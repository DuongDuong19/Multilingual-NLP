"""
model.py
========
Kiến trúc Multilingual NLP Model:

  ┌─────────────────────────────────────────────────────────┐
  │                   INPUT (text đa ngôn ngữ)              │
  └───────────────────────┬─────────────────────────────────┘
                          │
  ┌───────────────────────▼─────────────────────────────────┐
  │          ENCODER: mmBERT-small (frozen or fine-tuned)   │
  │    - 22 layers, hidden=384, ModernBERT architecture     │
  │    - Hỗ trợ 1800+ ngôn ngữ, max_len=8192               │
  │    Output: hidden_states [batch, seq_len, 384]          │
  └─────────────┬──────────────────────────────┬────────────┘
                │                              │
  ┌─────────────▼──────────┐    ┌─────────────▼────────────┐
  │     NER HEAD           │    │    PROJECTION LAYER      │
  │  Dropout + Linear      │    │    Linear(384 → 512)     │
  │  (384 → num_labels)    │    │    (align dims với mT5)  │
  │  Output: NER logits    │    └─────────────┬────────────┘
  │  [batch, seq_len, L]   │                  │
  └────────────────────────┘    ┌─────────────▼────────────┐
                                 │    DECODER: mT5-small    │
                                 │  Cross-attention vào     │
                                 │  encoder output          │
                                 │  Output: gen logits      │
                                 └──────────────────────────┘

TASK:
  - NER:         Token classification (tên người, tổ chức, địa điểm...)
  - Generation:  Seq2Seq (dịch thuật, tóm tắt, Q&A...)
  - Multi-task:  Kết hợp cả hai loss khi training
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoTokenizer,
    MT5ForConditionalGeneration,
    MT5Config,
)
from transformers.modeling_outputs import BaseModelOutput
from typing import Optional, Dict, Any, Tuple
import os

from config import Config, DEFAULT_CONFIG


class MultilingualNLPModel(nn.Module):
    """
    Model đa ngôn ngữ kết hợp:
      - mmBERT-small làm encoder (understanding)
      - NER head cho token classification
      - mT5-small decoder cho generation

    Cách dùng:
        model = MultilingualNLPModel(config)
        outputs = model.forward(input_ids, attention_mask, ner_labels=labels)
        outputs = model.forward(input_ids, attention_mask, decoder_input_ids=dec_ids, lm_labels=labels)
    """

    def __init__(self, config: Config = DEFAULT_CONFIG):
        super().__init__()
        self.config = config
        mc = config.model  # ModelConfig shortcut

        print(f"[Model] Đang load encoder: {mc.encoder_name}")
        # ----------------------------------------------------------------
        # 1. ENCODER: mmBERT-small
        #    AutoModel tự động nhận ra ModernBERT architecture
        # ----------------------------------------------------------------
        self.encoder = AutoModel.from_pretrained(
            mc.encoder_name,
            # add_pooling_layer=False sẽ trả về raw hidden states
        )
        encoder_dim = mc.encoder_hidden_size  # 384 với mmBERT-small

        # ----------------------------------------------------------------
        # 2. NER HEAD
        #    Dropout → Linear → [batch, seq_len, num_labels]
        # ----------------------------------------------------------------
        self.ner_dropout = nn.Dropout(mc.ner_dropout)
        self.ner_classifier = nn.Linear(encoder_dim, mc.num_ner_labels)

        # Loss function cho NER: ignore_index=-100 là padding/special tokens
        self.ner_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        # ----------------------------------------------------------------
        # 3. PROJECTION LAYER
        #    mmBERT-small dim (384) → mT5-small dim (512)
        #    Cần thiết vì cross-attention trong mT5 decoder
        #    expects key/value có cùng dim với mT5's hidden
        # ----------------------------------------------------------------
        decoder_dim = mc.decoder_hidden_size  # 512 với mT5-small
        self.encoder_projection = nn.Linear(encoder_dim, decoder_dim)

        # ----------------------------------------------------------------
        # 4. DECODER: mT5-small
        #    Load toàn bộ mT5, nhưng khi forward ta BYPASS encoder của nó
        #    bằng cách pass encoder_outputs trực tiếp
        # ----------------------------------------------------------------
        print(f"[Model] Đang load decoder: {mc.decoder_name}")
        self.mt5 = MT5ForConditionalGeneration.from_pretrained(mc.decoder_name)

        # Loss weights cho multi-task
        self.ner_loss_weight = mc.ner_loss_weight
        self.lm_loss_weight = mc.lm_loss_weight

        print(f"[Model] Model khởi tạo thành công!")
        print(f"  - Encoder dim: {encoder_dim}")
        print(f"  - Decoder dim: {decoder_dim}")
        print(f"  - NER labels: {mc.num_ner_labels}")
        self._print_param_count()

    def _print_param_count(self):
        """In số lượng parameters."""
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        ner_params = sum(p.numel() for p in self.ner_classifier.parameters())
        proj_params = sum(p.numel() for p in self.encoder_projection.parameters())
        mt5_params = sum(p.numel() for p in self.mt5.parameters())
        total = encoder_params + ner_params + proj_params + mt5_params
        print(f"  - mmBERT encoder:  {encoder_params/1e6:.1f}M params")
        print(f"  - NER head:        {ner_params/1e3:.1f}K params")
        print(f"  - Projection:      {proj_params/1e3:.1f}K params")
        print(f"  - mT5 decoder:     {mt5_params/1e6:.1f}M params")
        print(f"  - TOTAL:           {total/1e6:.1f}M params")

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Chỉ chạy encoder, trả về hidden states.

        Args:
            input_ids:      [batch, seq_len]
            attention_mask: [batch, seq_len]

        Returns:
            hidden_states: [batch, seq_len, 384]
        """
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # token_type_ids không dùng với mmBERT (ModernBERT)
        )
        # last_hidden_state: [batch, seq_len, hidden_size]
        return encoder_outputs.last_hidden_state

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        ner_labels: Optional[torch.Tensor] = None,   # [batch, seq_len] với giá trị label id
        lm_labels: Optional[torch.Tensor] = None,    # [batch, dec_seq_len] với token ids
    ) -> Dict[str, Any]:
        """
        FORWARD PASS chính.

        Args:
            input_ids:             [batch, seq_len] - token ids đầu vào
            attention_mask:        [batch, seq_len] - 1=real token, 0=padding
            decoder_input_ids:     [batch, dec_len] - decoder input (shifted lm_labels)
            decoder_attention_mask:[batch, dec_len] - mask cho decoder
            ner_labels:            [batch, seq_len] - ground truth NER labels
                                   -100 cho padding và special tokens (bị ignore)
            lm_labels:             [batch, dec_len] - ground truth cho LM
                                   -100 cho padding

        Returns:
            dict với các keys:
              - "total_loss":   Tổng loss (NER + LM), None nếu không có labels
              - "ner_loss":     NER loss riêng
              - "lm_loss":      Language model loss riêng
              - "ner_logits":   [batch, seq_len, num_labels]
              - "lm_logits":    [batch, dec_len, vocab_size], có khi None
        """

        # ================================================================
        # BƯỚC 1: ENCODE với mmBERT
        # ================================================================
        # hidden_states: [batch, seq_len, 384]
        hidden_states = self.encode(input_ids, attention_mask)

        # ================================================================
        # BƯỚC 2: NER HEAD
        # ================================================================
        # Dropout để regularize
        ner_hidden = self.ner_dropout(hidden_states)
        # Linear: [batch, seq_len, 384] → [batch, seq_len, num_labels]
        ner_logits = self.ner_classifier(ner_hidden)

        # Tính NER loss nếu có labels
        ner_loss = None
        if ner_labels is not None:
            # CrossEntropy expects [N, C] và [N]
            # Reshape: [batch*seq_len, num_labels] và [batch*seq_len]
            batch_size, seq_len, num_labels = ner_logits.shape
            ner_loss = self.ner_loss_fn(
                ner_logits.view(-1, num_labels),   # [batch*seq_len, num_labels]
                ner_labels.view(-1),               # [batch*seq_len]
            )

        # ================================================================
        # BƯỚC 3: PROJECT encoder output cho mT5 decoder
        # ================================================================
        # projected: [batch, seq_len, 384] → [batch, seq_len, 512]
        projected_hidden = self.encoder_projection(hidden_states)

        # Wrap thành BaseModelOutput mà mT5 decoder có thể đọc
        # mT5's decoder dùng encoder_hidden_states cho cross-attention
        encoder_outputs_for_mt5 = BaseModelOutput(
            last_hidden_state=projected_hidden,  # [batch, seq_len, 512]
        )

        # ================================================================
        # BƯỚC 4: mT5 DECODER
        # ================================================================
        lm_loss = None
        lm_logits = None

        # Chỉ chạy decoder nếu có decoder_input_ids hoặc lm_labels
        if decoder_input_ids is not None or lm_labels is not None:
            mt5_outputs = self.mt5(
                # Bypass mT5's encoder bằng cách pass encoder_outputs trực tiếp
                encoder_outputs=encoder_outputs_for_mt5,
                # attention_mask của encoder (cho cross-attention)
                attention_mask=attention_mask,
                # Input của decoder
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                # Labels: mT5 tự tính cross-entropy loss khi có labels
                labels=lm_labels,
                # Không cần return dict, mặc định là True trong transformers
                return_dict=True,
            )
            lm_loss = mt5_outputs.loss       # Scalar
            lm_logits = mt5_outputs.logits   # [batch, dec_len, vocab_size]

        # ================================================================
        # BƯỚC 5: TỔNG HỢP LOSS (Multi-task)
        # ================================================================
        total_loss = None
        if ner_loss is not None or lm_loss is not None:
            total_loss = torch.tensor(0.0, device=input_ids.device)
            if ner_loss is not None:
                total_loss = total_loss + self.ner_loss_weight * ner_loss
            if lm_loss is not None:
                total_loss = total_loss + self.lm_loss_weight * lm_loss

        return {
            "total_loss": total_loss,
            "ner_loss":   ner_loss,
            "lm_loss":    lm_loss,
            "ner_logits": ner_logits,    # [batch, seq_len, num_labels]
            "lm_logits":  lm_logits,     # [batch, dec_len, vocab_size] hoặc None
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 128,
        num_beams: int = 4,
        **generation_kwargs,
    ) -> torch.Tensor:
        """
        Sinh text với beam search thông qua mT5 decoder.

        Args:
            input_ids:      [batch, seq_len]
            attention_mask: [batch, seq_len]
            max_new_tokens: Độ dài tối đa output
            num_beams:      Số beam cho beam search

        Returns:
            generated_ids: [batch, output_len] - token ids đã sinh
        """
        # Encode trước
        hidden_states = self.encode(input_ids, attention_mask)
        projected = self.encoder_projection(hidden_states)
        encoder_outputs = BaseModelOutput(last_hidden_state=projected)

        # Generate với mT5
        generated = self.mt5.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            **generation_kwargs,
        )
        return generated

    @torch.no_grad()
    def predict_ner(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Dự đoán NER labels.

        Returns:
            predicted_ids: [batch, seq_len] - id của label được dự đoán
        """
        outputs = self.forward(input_ids, attention_mask)
        ner_logits = outputs["ner_logits"]  # [batch, seq_len, num_labels]
        return ner_logits.argmax(dim=-1)    # [batch, seq_len]

    def freeze_encoder(self):
        """Đóng băng encoder, chỉ train NER head và decoder."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("[Model] Encoder đã bị đóng băng (frozen).")

    def unfreeze_encoder(self):
        """Mở khóa encoder để fine-tune toàn bộ."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        print("[Model] Encoder đã được mở khóa (unfrozen).")

    def save(self, save_dir: str):
        """
        Lưu toàn bộ model.
        Lưu cả weights và cấu hình riêng biệt.

        Args:
            save_dir: Thư mục lưu model
        """
        os.makedirs(save_dir, exist_ok=True)

        # Lưu state dict của toàn bộ model
        model_path = os.path.join(save_dir, "model.pt")
        torch.save(self.state_dict(), model_path)

        # Lưu encoder riêng (để dùng lại với HuggingFace)
        encoder_path = os.path.join(save_dir, "encoder")
        self.encoder.save_pretrained(encoder_path)

        # Lưu mT5 riêng
        mt5_path = os.path.join(save_dir, "mt5")
        self.mt5.save_pretrained(mt5_path)

        # Lưu config dưới dạng json
        import json
        import dataclasses
        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(dataclasses.asdict(self.config), f, indent=2, ensure_ascii=False)

        print(f"[Model] Đã lưu model vào: {save_dir}")

    @classmethod
    def load(cls, save_dir: str, config: Optional[Config] = None) -> "MultilingualNLPModel":
        """
        Load model đã lưu.

        Args:
            save_dir:  Thư mục chứa model đã save
            config:    Config object (nếu None sẽ đọc từ config.json)

        Returns:
            model: MultilingualNLPModel đã load weights
        """
        import json

        if config is None:
            config_path = os.path.join(save_dir, "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    config_dict = json.load(f)
                # Tái tạo config từ dict
                from config import ModelConfig, TrainingConfig, DataConfig
                config = Config(
                    model=ModelConfig(**config_dict.get("model", {})),
                    training=TrainingConfig(**config_dict.get("training", {})),
                    data=DataConfig(**config_dict.get("data", {})),
                )
            else:
                print("[Model] Không tìm thấy config.json, dùng DEFAULT_CONFIG")
                config = DEFAULT_CONFIG

        # Tạo model với config
        model = cls(config)

        # Load weights
        model_path = os.path.join(save_dir, "model.pt")
        if os.path.exists(model_path):
            # map_location để load từ GPU sang CPU hoặc ngược lại
            state_dict = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=True)
            print(f"[Model] Đã load weights từ: {model_path}")
        else:
            print(f"[Model] CẢNH BÁO: Không tìm thấy {model_path}")

        return model