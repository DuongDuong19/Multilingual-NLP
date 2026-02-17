"""
File: src/config.py
Mô tả: Configuration cho training
"""

from dataclasses import dataclass

@dataclass
class ModelConfig:
    mbert_model_name: str = "bert-base-multilingual-cased"
    mt5_model_name: str = "google/mt5-small"
    max_length: int = 128

@dataclass
class TrainingConfig:
    batch_size: int = 8
    num_epochs: int = 3
    learning_rate: float = 5e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    use_amp: bool = True
    save_dir: str = "./models/checkpoints"
    log_dir: str = "./logs"

@dataclass
class DataConfig:
    dataset_name: str = "Helsinki-NLP/opus-100"   # ← đã sửa
    dataset_config: str = "en-vi"                  # ← đã sửa
    source_lang: str = "en"
    target_lang: str = "vi"
    min_length: int = 3
    max_length: int = 150
    num_workers: int = 2