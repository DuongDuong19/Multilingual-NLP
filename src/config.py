"""
File: src/config.py
Mô tả: Configuration cho training
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Model configuration"""
    mbert_model_name: str = 'bert-base-multilingual-cased'
    mt5_model_name: str = 'google/mt5-small'
    max_length: int = 128


@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 8
    num_epochs: int = 3
    learning_rate: float = 5e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    use_amp: bool = True  # Mixed precision training
    
    # Paths
    save_dir: str = './models/checkpoints'
    log_dir: str = './logs'
    
    # Evaluation
    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 100


@dataclass
class DataConfig:
    """Data configuration"""
    dataset_name: str = 'iwslt2017'
    dataset_config: str = 'iwslt2017-en-vi'
    min_length: int = 3
    max_length: int = 150
    num_workers: int = 2