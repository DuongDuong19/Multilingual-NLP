"""
main.py
=======
Entry point chính của Multilingual NLP Project.

Cách chạy:

  # Training mới:
  python main.py --mode train

  # Resume training từ checkpoint:
  python main.py --mode train --resume ./checkpoints/checkpoint_step_500_xxx

  # Evaluation trên test set:
  python main.py --mode eval --checkpoint ./checkpoints/checkpoint_best_xxx

  # Inference:
  python main.py --mode infer --checkpoint ./checkpoints/checkpoint_best_xxx \
                 --text "Barack Obama was born in Hawaii"

  # Tạo sample data để test:
  python main.py --mode create_data
"""

import argparse
import os
import sys
import torch
from transformers import AutoTokenizer

# Import các module trong project
from config import Config, ModelConfig, TrainingConfig, DataConfig, DEFAULT_CONFIG
from model import MultilingualNLPModel
from dataset import get_dataloaders, create_sample_data
from trainer import Trainer
from inference import InferencePipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multilingual NLP: mmBERT-small + NER + mT5-small",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        choices=["train", "eval", "infer", "create_data"],
        default="train",
        help="Chế độ chạy",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Đường dẫn checkpoint để tiếp tục training",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint để load cho eval/inference",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Text để inference",
    )
    parser.add_argument(
        "--task",
        choices=["ner", "generate", "analyze"],
        default="analyze",
        help="Task cho inference mode",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints",
        help="Thư mục lưu checkpoint",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Thư mục chứa data",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Số epochs training",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
        help="Đóng băng mmBERT encoder (chỉ train NER head và decoder)",
    )
    parser.add_argument(
        "--task_mode",
        choices=["ner", "seq2seq", "multitask"],
        default="multitask",
        help="Chế độ task training",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: cuda, cpu, mps (mặc định tự detect)",
    )

    return parser.parse_args()


def build_config(args) -> Config:
    """Xây dựng Config từ args."""
    config = Config(
        model=ModelConfig(),
        training=TrainingConfig(
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        ),
        data=DataConfig(
            train_file=os.path.join(args.data_dir, "train.json"),
            val_file=os.path.join(args.data_dir, "val.json"),
            test_file=os.path.join(args.data_dir, "test.json"),
        ),
    )
    return config


def mode_train(args):
    """Training mode."""
    print("\n" + "="*60)
    print("  MODE: TRAINING")
    print("="*60)

    config = build_config(args)
    config.create_dirs()

    # Kiểm tra data có tồn tại không
    for split in ["train", "val", "test"]:
        path = os.path.join(args.data_dir, f"{split}.json")
        if not os.path.exists(path):
            print(f"[Main] Không tìm thấy {path}")
            print(f"[Main] Chạy: python main.py --mode create_data  để tạo sample data")
            sys.exit(1)

    # ----------------------------------------------------------------
    # LOAD TOKENIZERS
    # ----------------------------------------------------------------
    mc = config.model
    print(f"\n[Main] Loading tokenizers...")

    # mmBERT dùng Gemma 2 tokenizer
    encoder_tokenizer = AutoTokenizer.from_pretrained(mc.encoder_name)
    # mT5 tokenizer
    decoder_tokenizer = AutoTokenizer.from_pretrained(mc.decoder_name)

    print(f"  Encoder vocab size: {encoder_tokenizer.vocab_size:,}")
    print(f"  Decoder vocab size: {decoder_tokenizer.vocab_size:,}")

    # ----------------------------------------------------------------
    # LOAD DATA
    # ----------------------------------------------------------------
    print(f"\n[Main] Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        config=config,
        encoder_tokenizer=encoder_tokenizer,
        decoder_tokenizer=decoder_tokenizer,
        task_mode=args.task_mode,
    )

    # ----------------------------------------------------------------
    # BUILD MODEL
    # ----------------------------------------------------------------
    print(f"\n[Main] Building model...")
    model = MultilingualNLPModel(config)

    # Freeze encoder nếu được yêu cầu (giúp training nhanh hơn, tránh catastrophic forgetting)
    if args.freeze_encoder:
        model.freeze_encoder()
        print("[Main] Encoder bị đóng băng. Chỉ train NER head + decoder + projection.")

    # ----------------------------------------------------------------
    # TRAIN
    # ----------------------------------------------------------------
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        id2label=mc.ner_id2label,
    )

    # Bắt đầu training (tự động resume nếu có checkpoint)
    trainer.train(resume_from=args.resume)

    # ----------------------------------------------------------------
    # EVALUATE TRÊN TEST SET SAU KHI TRAIN
    # ----------------------------------------------------------------
    print("\n[Main] Đánh giá trên test set...")
    test_metrics = trainer.evaluate(test_loader)
    print(f"  Test Loss:      {test_metrics['total_loss']:.4f}")
    print(f"  Test NER F1:    {test_metrics['ner_f1']:.4f}")
    print(f"  Test Precision: {test_metrics['ner_precision']:.4f}")
    print(f"  Test Recall:    {test_metrics['ner_recall']:.4f}")


def mode_eval(args):
    """Evaluation mode."""
    print("\n" + "="*60)
    print("  MODE: EVALUATION")
    print("="*60)

    if args.checkpoint is None:
        print("[Main] Cần cung cấp --checkpoint để evaluate")
        sys.exit(1)

    config = build_config(args)

    # Load model
    model = MultilingualNLPModel.load(args.checkpoint, config)
    model.eval()

    # Load tokenizers
    mc = config.model
    encoder_tokenizer = AutoTokenizer.from_pretrained(mc.encoder_name)
    decoder_tokenizer = AutoTokenizer.from_pretrained(mc.decoder_name)

    # Load test data
    _, _, test_loader = get_dataloaders(config, encoder_tokenizer, decoder_tokenizer)

    # Evaluate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Dùng Trainer chỉ để evaluate (không cần optimizer)
    from torch.utils.data import DataLoader
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=test_loader,  # Dummy, không dùng
        val_loader=test_loader,
        id2label=mc.ner_id2label,
    )
    metrics = trainer.evaluate(test_loader)

    print(f"\nKết quả đánh giá:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


def mode_infer(args):
    """Inference mode."""
    print("\n" + "="*60)
    print("  MODE: INFERENCE")
    print("="*60)

    if args.checkpoint is None:
        print("[Main] Cần cung cấp --checkpoint để inference")
        sys.exit(1)

    if args.text is None:
        # Demo với text mẫu
        args.text = "Barack Obama was the 44th President of the United States, born in Hawaii."
        print(f"[Main] Dùng text mẫu: {args.text}")

    # Load pipeline
    pipeline = InferencePipeline.from_checkpoint(args.checkpoint)

    print(f"\nInput: {args.text}")
    print("-" * 40)

    if args.task == "ner":
        # NER prediction
        entities = pipeline.extract_entities(args.text)
        print("Entities được phát hiện:")
        for e in entities:
            print(f"  [{e['label']}] '{e['text']}' (confidence: {e['score']:.3f})")

    elif args.task == "generate":
        # Text generation
        output = pipeline.generate(args.text)
        print(f"Generated: {output}")

    else:  # analyze
        # Phân tích toàn diện
        result = pipeline.analyze(args.text)
        print("Entities:")
        for e in result["entities"]:
            print(f"  [{e['label']}] '{e['text']}' (score: {e['score']:.3f})")
        print(f"\nGenerated text: {result['generated_text']}")


def mode_create_data(args):
    """Tạo sample data để test."""
    print("\n[Main] Tạo sample data...")
    create_sample_data(args.data_dir)
    print(f"\nDone! Data được tạo tại: {args.data_dir}/")
    print("Bây giờ có thể chạy: python main.py --mode train")


# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    args = parse_args()

    print("\n" + "="*60)
    print("  Multilingual NLP: mmBERT-small + NER + mT5-small")
    print("="*60)
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    if args.mode == "train":
        mode_train(args)
    elif args.mode == "eval":
        mode_eval(args)
    elif args.mode == "infer":
        mode_infer(args)
    elif args.mode == "create_data":
        mode_create_data(args)
    else:
        print(f"Mode không hợp lệ: {args.mode}")
        sys.exit(1)