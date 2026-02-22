"""
config.py
=========
Tất cả cấu hình của project Multilingual-NLP.
Thay đổi các giá trị ở đây thay vì sửa code trong các file khác.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


# ================================================================
# CẤU HÌNH MODEL
# ================================================================
@dataclass
class ModelConfig:
    # --- Encoder: mmBERT-small ---
    encoder_name: str = "jhu-clsp/mmBERT-small"
    # Hidden size của mmBERT-small = 384
    encoder_hidden_size: int = 384
    # Max sequence length mmBERT-small hỗ trợ = 8192
    max_encoder_length: int = 512      # Dùng 512 cho phù hợp GPU nhỏ

    # --- Decoder: mT5-small ---
    decoder_name: str = "google/mt5-small"
    # Hidden size của mT5-small = 512
    decoder_hidden_size: int = 512
    # Max length của output sequence (generation)
    max_decoder_length: int = 128

    # --- NER Head ---
    # Danh sách nhãn NER, thay đổi theo dataset của bạn
    # Đây là ví dụ với CoNLL-2003 style (BIO scheme)
    ner_labels: List[str] = field(default_factory=lambda: [
        "O",          # Outside (không phải entity)
        "B-PER",      # Beginning - Person
        "I-PER",      # Inside - Person
        "B-ORG",      # Beginning - Organization
        "I-ORG",      # Inside - Organization
        "B-LOC",      # Beginning - Location
        "I-LOC",      # Inside - Location
        "B-MISC",     # Beginning - Miscellaneous
        "I-MISC",     # Inside - Miscellaneous
    ])
    ner_dropout: float = 0.1   # Dropout trước NER classifier

    # --- Loss weights (multi-task learning) ---
    # Tổng loss = ner_loss_weight * NER_loss + lm_loss_weight * LM_loss
    ner_loss_weight: float = 1.0
    lm_loss_weight: float = 1.0

    @property
    def num_ner_labels(self) -> int:
        return len(self.ner_labels)

    @property
    def ner_label2id(self) -> dict:
        return {label: i for i, label in enumerate(self.ner_labels)}

    @property
    def ner_id2label(self) -> dict:
        return {i: label for i, label in enumerate(self.ner_labels)}


# ================================================================
# CẤU HÌNH TRAINING
# ================================================================
@dataclass
class TrainingConfig:
    # --- Paths ---
    output_dir: str = "./checkpoints"       # Lưu checkpoint vào đây
    log_dir: str = "./logs"                 # TensorBoard logs
    best_model_dir: str = "./best_model"    # Lưu model tốt nhất

    # --- Training hyperparameters ---
    num_epochs: int = 10
    batch_size: int = 16                   # Điều chỉnh theo GPU memory
    gradient_accumulation_steps: int = 4   # Effective batch = 16 * 4 = 64
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1             # 10% steps đầu là warmup
    max_grad_norm: float = 1.0            # Gradient clipping

    # --- Mixed precision ---
    # "fp16" hoặc "bf16" hoặc "no"
    mixed_precision: str = "bf16"         # bf16 tốt hơn fp16 trên A100/H100

    # --- Auto-save mỗi 30 phút ---
    autosave_interval_minutes: int = 30   # Lưu checkpoint tự động mỗi 30 phút
    save_total_limit: int = 3             # Giữ tối đa 3 checkpoint gần nhất

    # --- Evaluation ---
    eval_steps: int = 500                 # Evaluate mỗi N steps
    logging_steps: int = 50              # Log mỗi N steps

    # --- Device ---
    device: str = "cuda"                 # "cuda" hoặc "cpu" hoặc "mps"
    dataloader_num_workers: int = 4

    # --- Seed ---
    seed: int = 42

    # --- Early stopping ---
    patience: int = 5                    # Dừng nếu không cải thiện sau N eval
    early_stopping_metric: str = "ner_f1"  # Metric để early stopping


# ================================================================
# CẤU HÌNH DATA
# ================================================================
@dataclass
class DataConfig:
    # Path đến data của bạn
    train_file: str = "./data/train.json"
    val_file: str = "./data/val.json"
    test_file: str = "./data/test.json"

    # Format: "json" hoặc "conll" (cho NER)
    data_format: str = "json"

    # Tỉ lệ split nếu chỉ có 1 file
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1


# ================================================================
# CONFIG TỔNG HỢP
# ================================================================
@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def create_dirs(self):
        """Tạo tất cả các thư mục cần thiết."""
        for path in [
            self.training.output_dir,
            self.training.log_dir,
            self.training.best_model_dir,
            os.path.dirname(self.data.train_file),
        ]:
            os.makedirs(path, exist_ok=True)


# Instance mặc định - import từ đây trong các file khác
DEFAULT_CONFIG = Config()