"""
File: scripts/evaluate.py
Mô tả: Evaluate trained model
"""

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
import sys
sys.path.append('..')

from src.model import MultilingualNERTranslationModel
from src.data_utils import TranslationDataset, NERTranslationDataCollator
from src.trainer import NERTranslationTrainer


def main():
    # Load test dataset
    print("Loading test dataset...")
    test_dataset = TranslationDataset(load_from_disk('./data/processed/test'))
    
    # Load model
    print("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultilingualNERTranslationModel()
    model.load_model('./models/best_model')
    model = model.to(device)
    
    # Create dataloader
    data_collator = NERTranslationDataCollator(model=model)
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=data_collator
    )
    
    # Evaluate
    print("Evaluating...")
    trainer = NERTranslationTrainer(
        model=model,
        train_loader=None,
        val_loader=None,
        optimizer=None,
        scheduler=None,
        device=device
    )
    
    bleu_score, predictions, references = trainer.compute_bleu(
        test_loader, 
        num_samples=len(test_dataset)
    )
    
    print(f"\nBLEU Score: {bleu_score:.2f}")
    
    # Show examples
    print("\nSample Translations:")
    for i in range(min(5, len(predictions))):
        print(f"\nPrediction: {predictions[i]}")
        print(f"Reference:  {references[i][0]}")


if __name__ == '__main__':
    main()