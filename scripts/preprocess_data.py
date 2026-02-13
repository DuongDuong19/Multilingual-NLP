"""
File: scripts/preprocess_data.py
Mô tả: Preprocess dataset
"""

from datasets import load_dataset
import sys
sys.path.append('..')
from src.data_utils import prepare_dataset


def main():
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("iwslt2017", "iwslt2017-en-vi", trust_remote_code=True)
    
    # Prepare dataset
    print("Preparing dataset...")
    dataset = prepare_dataset(dataset, min_len=3, max_len=150)
    
    # Save to disk
    print("Saving dataset...")
    dataset['train'].save_to_disk('./data/processed/train')
    dataset['validation'].save_to_disk('./data/processed/validation')
    dataset['test'].save_to_disk('./data/processed/test')
    
    print("Done!")
    print(f"Train samples: {len(dataset['train'])}")
    print(f"Validation samples: {len(dataset['validation'])}")
    print(f"Test samples: {len(dataset['test'])}")


if __name__ == '__main__':
    main()