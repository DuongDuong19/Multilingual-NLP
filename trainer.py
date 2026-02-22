"""
trainer.py
==========
Training loop cho Multilingual NLP Model.

Tính năng chính:
  ✓ Forward + Backward pass
  ✓ Gradient clipping
  ✓ Mixed precision (bf16/fp16)
  ✓ Gradient accumulation
  ✓ Auto-save mỗi 30 phút (tránh crash mất công)
  ✓ Save/Load checkpoint đầy đủ (model + optimizer + scheduler + trạng thái)
  ✓ Early stopping
  ✓ Evaluation với NER metrics (F1, Precision, Recall)
  ✓ TensorBoard logging
"""

import os
import time
import json
import glob
import shutil
from typing import Dict, Optional, Tuple, List
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

from tqdm import tqdm
import numpy as np

try:
    from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
    SEQEVAL_AVAILABLE = True
except ImportError:
    SEQEVAL_AVAILABLE = False
    print("[Trainer] CẢNH BÁO: seqeval chưa cài. Chạy: pip install seqeval")

try:
    from torch.utils.tensorboard import SummaryWriter
    TB_AVAILABLE = True
except ImportError:
    TB_AVAILABLE = False

from config import Config, DEFAULT_CONFIG
from model import MultilingualNLPModel


class CheckpointManager:
    """
    Quản lý việc lưu và load checkpoint.
    Giữ tối đa N checkpoint gần nhất để tiết kiệm disk.
    """

    def __init__(self, output_dir: str, save_total_limit: int = 3):
        self.output_dir = output_dir
        self.save_total_limit = save_total_limit
        os.makedirs(output_dir, exist_ok=True)

    def save(
        self,
        model: MultilingualNLPModel,
        optimizer: torch.optim.Optimizer,
        scheduler,
        scaler: Optional[GradScaler],
        step: int,
        epoch: int,
        metrics: Dict,
        tag: str = "step",  # "step", "best", "autosave"
    ) -> str:
        """
        Lưu checkpoint đầy đủ.

        Returns:
            checkpoint_dir: Đường dẫn thư mục checkpoint vừa lưu
        """
        # Tên thư mục checkpoint
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_{tag}_{step}_{timestamp}"
        checkpoint_dir = os.path.join(self.output_dir, checkpoint_name)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # 1. Lưu model weights (dùng method của model)
        model.save(checkpoint_dir)

        # 2. Lưu trạng thái optimizer
        torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))

        # 3. Lưu scheduler
        torch.save(scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pt"))

        # 4. Lưu scaler (cho mixed precision)
        if scaler is not None:
            torch.save(scaler.state_dict(), os.path.join(checkpoint_dir, "scaler.pt"))

        # 5. Lưu metadata (step, epoch, metrics...)
        metadata = {
            "step": step,
            "epoch": epoch,
            "metrics": {k: float(v) if v is not None else None for k, v in metrics.items()},
            "timestamp": timestamp,
            "tag": tag,
        }
        with open(os.path.join(checkpoint_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"[Checkpoint] ✓ Đã lưu: {checkpoint_name}")

        # Xóa checkpoint cũ nếu vượt giới hạn
        # Không xóa checkpoint "best"
        if tag != "best":
            self._cleanup_old_checkpoints()

        return checkpoint_dir

    def _cleanup_old_checkpoints(self):
        """Xóa checkpoint cũ, giữ lại save_total_limit checkpoint gần nhất."""
        # Lấy tất cả checkpoint (trừ "best")
        all_checkpoints = sorted(
            glob.glob(os.path.join(self.output_dir, "checkpoint_step_*")),
        )
        all_checkpoints += sorted(
            glob.glob(os.path.join(self.output_dir, "checkpoint_autosave_*")),
        )
        # Sort theo thời gian tạo
        all_checkpoints.sort(key=lambda x: os.path.getctime(x))

        # Xóa checkpoint cũ nhất nếu vượt limit
        while len(all_checkpoints) > self.save_total_limit:
            oldest = all_checkpoints.pop(0)
            shutil.rmtree(oldest)
            print(f"[Checkpoint] Đã xóa checkpoint cũ: {os.path.basename(oldest)}")

    def load(
        self,
        checkpoint_dir: str,
        model: MultilingualNLPModel,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler=None,
        scaler: Optional[GradScaler] = None,
    ) -> Dict:
        """
        Load checkpoint.

        Returns:
            metadata: Dict chứa step, epoch, metrics...
        """
        # Load model weights
        model_path = os.path.join(checkpoint_dir, "model.pt")
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state_dict)
            print(f"[Checkpoint] Đã load model từ {checkpoint_dir}")

        # Load optimizer
        opt_path = os.path.join(checkpoint_dir, "optimizer.pt")
        if optimizer is not None and os.path.exists(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))

        # Load scheduler
        sch_path = os.path.join(checkpoint_dir, "scheduler.pt")
        if scheduler is not None and os.path.exists(sch_path):
            scheduler.load_state_dict(torch.load(sch_path, map_location="cpu"))

        # Load scaler
        scl_path = os.path.join(checkpoint_dir, "scaler.pt")
        if scaler is not None and os.path.exists(scl_path):
            scaler.load_state_dict(torch.load(scl_path, map_location="cpu"))

        # Load metadata
        meta_path = os.path.join(checkpoint_dir, "metadata.json")
        metadata = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                metadata = json.load(f)

        return metadata

    def find_latest_checkpoint(self) -> Optional[str]:
        """Tìm checkpoint mới nhất trong output_dir."""
        checkpoints = glob.glob(os.path.join(self.output_dir, "checkpoint_*"))
        if not checkpoints:
            return None
        # Sort theo thời gian tạo, lấy mới nhất
        latest = max(checkpoints, key=os.path.getctime)
        return latest

    def find_best_checkpoint(self) -> Optional[str]:
        """Tìm checkpoint được đánh dấu là best."""
        best_dir = os.path.join(self.output_dir, "checkpoint_best_*")
        checkpoints = glob.glob(best_dir)
        if not checkpoints:
            return None
        return max(checkpoints, key=os.path.getctime)


class Trainer:
    """
    Trainer chính cho MultilingualNLPModel.

    Ví dụ sử dụng:
        trainer = Trainer(model, config, train_loader, val_loader)
        trainer.train()
    """

    def __init__(
        self,
        model: MultilingualNLPModel,
        config: Config,
        train_loader: DataLoader,
        val_loader: DataLoader,
        id2label: Dict[int, str],
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.id2label = id2label
        tc = config.training

        # ----------------------------------------------------------------
        # DEVICE
        # ----------------------------------------------------------------
        if tc.device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"[Trainer] Dùng GPU: {torch.cuda.get_device_name(0)}")
        elif tc.device == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("[Trainer] Dùng Apple MPS")
        else:
            self.device = torch.device("cpu")
            print("[Trainer] Dùng CPU (chậm, chỉ dùng để test)")

        self.model = self.model.to(self.device)

        # ----------------------------------------------------------------
        # OPTIMIZER
        # AdamW với weight decay, không áp dụng weight decay cho bias/norm
        # ----------------------------------------------------------------
        # Tách parameters: có/không weight decay
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        param_groups = [
            {
                "params": [p for n, p in model.named_parameters()
                           if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": tc.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters()
                           if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(param_groups, lr=tc.learning_rate)

        # ----------------------------------------------------------------
        # SCHEDULER: Warmup → Cosine decay
        # ----------------------------------------------------------------
        total_steps = len(train_loader) * tc.num_epochs // tc.gradient_accumulation_steps
        warmup_steps = int(total_steps * tc.warmup_ratio)

        # Phase 1: Linear warmup
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,    # Bắt đầu từ lr * 0.1
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        # Phase 2: Cosine decay
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=tc.learning_rate * 0.01,  # LR tối thiểu = 1% của lr ban đầu
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )

        print(f"[Trainer] Total steps: {total_steps} | Warmup steps: {warmup_steps}")

        # ----------------------------------------------------------------
        # MIXED PRECISION SCALER
        # ----------------------------------------------------------------
        self.use_amp = tc.mixed_precision in ("fp16", "bf16") and self.device.type == "cuda"
        self.amp_dtype = torch.bfloat16 if tc.mixed_precision == "bf16" else torch.float16
        # GradScaler chỉ cần với fp16, không cần với bf16
        self.scaler = GradScaler() if (self.use_amp and tc.mixed_precision == "fp16") else None

        # ----------------------------------------------------------------
        # CHECKPOINT MANAGER
        # ----------------------------------------------------------------
        self.checkpoint_manager = CheckpointManager(
            output_dir=tc.output_dir,
            save_total_limit=tc.save_total_limit,
        )

        # ----------------------------------------------------------------
        # TENSORBOARD
        # ----------------------------------------------------------------
        self.writer = None
        if TB_AVAILABLE:
            os.makedirs(tc.log_dir, exist_ok=True)
            self.writer = SummaryWriter(tc.log_dir)

        # ----------------------------------------------------------------
        # TRACKING VARIABLES
        # ----------------------------------------------------------------
        self.global_step = 0
        self.current_epoch = 0
        self.best_metric = 0.0       # Cao hơn = tốt hơn
        self.no_improve_count = 0    # Đếm số lần eval không cải thiện

        # Auto-save timing
        self.last_autosave_time = time.time()
        self.autosave_interval = tc.autosave_interval_minutes * 60  # Chuyển sang giây

        # Set random seed
        self._set_seed(tc.seed)

    def _set_seed(self, seed: int):
        """Set seed cho reproducibility."""
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Chuyển tất cả tensors trong batch sang device."""
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

    def train_step(self, batch: Dict) -> Dict[str, float]:
        """
        FORWARD + BACKWARD cho 1 batch.

        Returns:
            losses: Dict chứa các loss values
        """
        self.model.train()
        batch = self._move_batch_to_device(batch)

        # ----------------------------------------------------------------
        # FORWARD PASS (với mixed precision nếu bật)
        # ----------------------------------------------------------------
        if self.use_amp:
            with autocast(device_type="cuda", dtype=self.amp_dtype):
                outputs = self.model(**batch)
        else:
            outputs = self.model(**batch)

        total_loss = outputs["total_loss"]
        if total_loss is None:
            raise ValueError("Batch không có labels để tính loss!")

        # Scale loss cho gradient accumulation
        tc = self.config.training
        scaled_loss = total_loss / tc.gradient_accumulation_steps

        # ----------------------------------------------------------------
        # BACKWARD PASS
        # ----------------------------------------------------------------
        if self.scaler is not None:
            # fp16: dùng scaler để tránh underflow
            self.scaler.scale(scaled_loss).backward()
        else:
            # bf16 hoặc fp32: backward bình thường
            scaled_loss.backward()

        losses = {
            "total_loss": total_loss.item(),
            "ner_loss": outputs["ner_loss"].item() if outputs["ner_loss"] is not None else 0.0,
            "lm_loss": outputs["lm_loss"].item() if outputs["lm_loss"] is not None else 0.0,
        }
        return losses

    def optimizer_step(self):
        """
        Clip gradient + update optimizer + scheduler.
        Gọi sau mỗi gradient_accumulation_steps bước.
        """
        tc = self.config.training

        if self.scaler is not None:
            # fp16: unscale trước khi clip
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), tc.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), tc.max_grad_norm)
            self.optimizer.step()

        self.scheduler.step()
        self.optimizer.zero_grad()

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Đánh giá model trên một dataloader.

        Returns:
            metrics: Dict với total_loss, ner_loss, lm_loss, ner_f1, ner_precision, ner_recall
        """
        self.model.eval()
        total_losses = {"total_loss": 0.0, "ner_loss": 0.0, "lm_loss": 0.0}
        all_predictions = []   # Cho NER
        all_true_labels = []   # Cho NER
        num_batches = 0

        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            batch = self._move_batch_to_device(batch)

            if self.use_amp:
                with autocast(device_type="cuda", dtype=self.amp_dtype):
                    outputs = self.model(**batch)
            else:
                outputs = self.model(**batch)

            # Tính loss
            if outputs["total_loss"] is not None:
                total_losses["total_loss"] += outputs["total_loss"].item()
            if outputs["ner_loss"] is not None:
                total_losses["ner_loss"] += outputs["ner_loss"].item()
            if outputs["lm_loss"] is not None:
                total_losses["lm_loss"] += outputs["lm_loss"].item()

            # Thu thập NER predictions để tính F1
            if outputs["ner_logits"] is not None and "ner_labels" in batch:
                ner_logits = outputs["ner_logits"]    # [batch, seq_len, num_labels]
                ner_labels = batch["ner_labels"]       # [batch, seq_len]
                predictions = ner_logits.argmax(dim=-1)  # [batch, seq_len]

                # Chuyển sang list of list of label strings, bỏ -100
                for pred_seq, label_seq in zip(predictions, ner_labels):
                    pred_labels_str = []
                    true_labels_str = []
                    for pred_id, true_id in zip(pred_seq.tolist(), label_seq.tolist()):
                        if true_id == -100:
                            continue  # Bỏ qua padding và special tokens
                        pred_labels_str.append(self.id2label.get(pred_id, "O"))
                        true_labels_str.append(self.id2label.get(true_id, "O"))
                    if pred_labels_str:
                        all_predictions.append(pred_labels_str)
                        all_true_labels.append(true_labels_str)

            num_batches += 1

        # Average losses
        metrics = {k: v / max(num_batches, 1) for k, v in total_losses.items()}

        # NER metrics với seqeval
        if SEQEVAL_AVAILABLE and all_predictions:
            metrics["ner_f1"] = f1_score(all_true_labels, all_predictions)
            metrics["ner_precision"] = precision_score(all_true_labels, all_predictions)
            metrics["ner_recall"] = recall_score(all_true_labels, all_predictions)
        else:
            metrics["ner_f1"] = 0.0
            metrics["ner_precision"] = 0.0
            metrics["ner_recall"] = 0.0

        return metrics

    def _should_autosave(self) -> bool:
        """Kiểm tra có nên auto-save không (dựa trên thời gian)."""
        elapsed = time.time() - self.last_autosave_time
        return elapsed >= self.autosave_interval

    def _log_metrics(self, metrics: Dict, prefix: str = "train"):
        """Log metrics lên TensorBoard và console."""
        if self.writer is not None:
            for k, v in metrics.items():
                if v is not None:
                    self.writer.add_scalar(f"{prefix}/{k}", v, self.global_step)

    def train(self, resume_from: Optional[str] = None):
        """
        Training loop chính.

        Args:
            resume_from: Đường dẫn checkpoint để tiếp tục training.
                         Nếu None, tự động tìm checkpoint mới nhất.
        """
        tc = self.config.training

        # ----------------------------------------------------------------
        # RESUME từ checkpoint nếu có
        # ----------------------------------------------------------------
        start_epoch = 0
        if resume_from is None:
            # Tự động tìm checkpoint mới nhất
            resume_from = self.checkpoint_manager.find_latest_checkpoint()

        if resume_from is not None and os.path.exists(resume_from):
            print(f"\n[Trainer] Resume từ checkpoint: {resume_from}")
            metadata = self.checkpoint_manager.load(
                resume_from, self.model, self.optimizer, self.scheduler, self.scaler
            )
            self.global_step = metadata.get("step", 0)
            start_epoch = metadata.get("epoch", 0)
            self.best_metric = metadata.get("metrics", {}).get(tc.early_stopping_metric, 0.0)
            print(f"[Trainer] Tiếp tục từ epoch {start_epoch}, step {self.global_step}")
        else:
            print("\n[Trainer] Bắt đầu training từ đầu.")

        self.model = self.model.to(self.device)

        print(f"\n{'='*60}")
        print(f"  Training: {tc.num_epochs} epochs | LR: {tc.learning_rate}")
        print(f"  Batch size: {tc.batch_size} | Grad accum: {tc.gradient_accumulation_steps}")
        print(f"  Auto-save mỗi: {tc.autosave_interval_minutes} phút")
        print(f"  Mixed precision: {tc.mixed_precision}")
        print(f"{'='*60}\n")

        # ----------------------------------------------------------------
        # TRAINING LOOP
        # ----------------------------------------------------------------
        for epoch in range(start_epoch, tc.num_epochs):
            self.current_epoch = epoch
            epoch_losses = {"total_loss": 0.0, "ner_loss": 0.0, "lm_loss": 0.0}
            num_steps_in_epoch = 0

            self.optimizer.zero_grad()  # Reset gradient ở đầu mỗi epoch

            # Progress bar cho epoch
            pbar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch+1}/{tc.num_epochs}",
                dynamic_ncols=True,
            )

            for step, batch in enumerate(pbar):
                # --------------------------------------------------------
                # FORWARD + BACKWARD
                # --------------------------------------------------------
                losses = self.train_step(batch)

                # Cộng dồn loss để log
                for k in epoch_losses:
                    epoch_losses[k] += losses.get(k, 0.0)
                num_steps_in_epoch += 1

                # --------------------------------------------------------
                # OPTIMIZER STEP (mỗi gradient_accumulation_steps bước)
                # --------------------------------------------------------
                is_accumulation_step = (step + 1) % tc.gradient_accumulation_steps == 0
                is_last_step = (step + 1) == len(self.train_loader)

                if is_accumulation_step or is_last_step:
                    self.optimizer_step()
                    self.global_step += 1

                    # Update progress bar
                    current_lr = self.scheduler.get_last_lr()[0]
                    pbar.set_postfix({
                        "loss": f"{losses['total_loss']:.4f}",
                        "ner": f"{losses['ner_loss']:.4f}",
                        "lm": f"{losses['lm_loss']:.4f}",
                        "lr": f"{current_lr:.2e}",
                    })

                    # Log to TensorBoard
                    if self.global_step % tc.logging_steps == 0:
                        train_metrics = {
                            **losses,
                            "learning_rate": current_lr,
                        }
                        self._log_metrics(train_metrics, prefix="train")

                    # --------------------------------------------------------
                    # EVALUATION (mỗi eval_steps)
                    # --------------------------------------------------------
                    if self.global_step % tc.eval_steps == 0:
                        print(f"\n[Eval] Step {self.global_step}...")
                        val_metrics = self.evaluate(self.val_loader)
                        self._log_metrics(val_metrics, prefix="val")

                        print(f"  Val Loss:      {val_metrics['total_loss']:.4f}")
                        print(f"  NER F1:        {val_metrics['ner_f1']:.4f}")
                        print(f"  NER Precision: {val_metrics['ner_precision']:.4f}")
                        print(f"  NER Recall:    {val_metrics['ner_recall']:.4f}")

                        # Kiểm tra có cải thiện không
                        current_metric = val_metrics.get(tc.early_stopping_metric, 0.0)
                        if current_metric > self.best_metric:
                            self.best_metric = current_metric
                            self.no_improve_count = 0
                            # Lưu model tốt nhất
                            self.checkpoint_manager.save(
                                model=self.model,
                                optimizer=self.optimizer,
                                scheduler=self.scheduler,
                                scaler=self.scaler,
                                step=self.global_step,
                                epoch=epoch,
                                metrics=val_metrics,
                                tag="best",
                            )
                            print(f"  ★ Best model! {tc.early_stopping_metric}: {current_metric:.4f}")
                        else:
                            self.no_improve_count += 1
                            print(f"  Không cải thiện ({self.no_improve_count}/{tc.patience})")

                        # Early stopping
                        if self.no_improve_count >= tc.patience:
                            print(f"\n[Trainer] Early stopping! "
                                  f"Không cải thiện sau {tc.patience} lần eval.")
                            return

                    # --------------------------------------------------------
                    # AUTO-SAVE MỖI 30 PHÚT
                    # --------------------------------------------------------
                    if self._should_autosave():
                        print(f"\n[AutoSave] ⏰ Đã qua {tc.autosave_interval_minutes} phút, lưu checkpoint...")
                        self.checkpoint_manager.save(
                            model=self.model,
                            optimizer=self.optimizer,
                            scheduler=self.scheduler,
                            scaler=self.scaler,
                            step=self.global_step,
                            epoch=epoch,
                            metrics={"note": "autosave"},
                            tag="autosave",
                        )
                        self.last_autosave_time = time.time()

            # ----------------------------------------------------------------
            # CUỐI MỖI EPOCH: Log + Save
            # ----------------------------------------------------------------
            avg_losses = {k: v / max(num_steps_in_epoch, 1) for k, v in epoch_losses.items()}
            print(f"\n[Epoch {epoch+1}] Avg Train Loss: {avg_losses['total_loss']:.4f}")

            # Save checkpoint cuối epoch
            val_metrics = self.evaluate(self.val_loader)
            self.checkpoint_manager.save(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                step=self.global_step,
                epoch=epoch + 1,
                metrics=val_metrics,
                tag="step",
            )

        print(f"\n[Trainer] Training hoàn thành! Best {tc.early_stopping_metric}: {self.best_metric:.4f}")

        # Đóng TensorBoard writer
        if self.writer is not None:
            self.writer.close()