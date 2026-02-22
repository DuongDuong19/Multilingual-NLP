"""
dataset.py
==========
Xử lý dữ liệu cho Multilingual NLP Model.

Hỗ trợ 2 kiểu task:
  1. NER Task    - Token classification
  2. Seq2Seq Task - Generation (dịch thuật, tóm tắt...)
  3. Multi-task  - Kết hợp cả hai

Format dữ liệu JSON ví dụ:

  NER:
  {
    "text": "Barack Obama was born in Hawaii.",
    "tokens": ["Barack", "Obama", "was", "born", "in", "Hawaii", "."],
    "ner_tags": ["B-PER", "I-PER", "O", "O", "O", "B-LOC", "O"]
  }

  Seq2Seq:
  {
    "source": "Hello, how are you?",
    "target": "Xin chào, bạn khỏe không?"
  }

  Multi-task:
  {
    "tokens": ["John", "went", "to", "Paris"],
    "ner_tags": ["B-PER", "O", "O", "B-LOC"],
    "source": "John went to Paris",
    "target": "John đã đến Paris"
  }
"""

import json
import os
from typing import List, Dict, Optional, Tuple, Any
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import AutoTokenizer, PreTrainedTokenizer

from config import Config, DEFAULT_CONFIG


class MultilingualNERDataset(Dataset):
    """
    Dataset cho NER task.

    Xử lý word-level labels → subword-level labels.
    Với mmBERT dùng Gemma 2 tokenizer, một word có thể thành nhiều subword.
    Chỉ label subword đầu tiên của mỗi word, phần còn lại là -100 (ignored).
    """

    def __init__(
        self,
        data: List[Dict],
        tokenizer: PreTrainedTokenizer,
        label2id: Dict[str, int],
        max_length: int = 512,
    ):
        """
        Args:
            data:       List of dicts, mỗi dict có "tokens" và "ner_tags"
            tokenizer:  HuggingFace tokenizer (của mmBERT)
            label2id:   Dict mapping label string → int
            max_length: Max sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        tokens: List[str] = item["tokens"]
        ner_tags: List[str] = item["ner_tags"]

        # Chuyển ner_tags sang ids
        word_labels = [self.label2id.get(tag, 0) for tag in ner_tags]

        # ----------------------------------------------------------------
        # Tokenize với word_ids để map subword → word
        # ----------------------------------------------------------------
        # is_split_into_words=True: đầu vào là list of words
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)       # [seq_len]
        attention_mask = encoding["attention_mask"].squeeze(0)  # [seq_len]

        # ----------------------------------------------------------------
        # Align labels với subwords
        # word_ids() trả về danh sách: None cho special tokens,
        # word_index cho mỗi subword
        # ----------------------------------------------------------------
        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels = []
        prev_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                # Special token ([CLS], [SEP], [PAD]) → ignore
                aligned_labels.append(-100)
            elif word_idx != prev_word_idx:
                # Subword ĐẦU TIÊN của word → dùng label thật
                aligned_labels.append(word_labels[word_idx] if word_idx < len(word_labels) else -100)
            else:
                # Subword tiếp theo của cùng word → ignore (-100)
                aligned_labels.append(-100)
            prev_word_idx = word_idx

        # Padding để đủ max_length
        while len(aligned_labels) < self.max_length:
            aligned_labels.append(-100)
        aligned_labels = aligned_labels[:self.max_length]

        labels = torch.tensor(aligned_labels, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "ner_labels": labels,
        }


class MultilingualSeq2SeqDataset(Dataset):
    """
    Dataset cho Seq2Seq task (dịch thuật, tóm tắt...).

    Encoder: tokenize source text với encoder tokenizer (mmBERT)
    Decoder: tokenize target text với decoder tokenizer (mT5)
    """

    def __init__(
        self,
        data: List[Dict],
        encoder_tokenizer: PreTrainedTokenizer,
        decoder_tokenizer: PreTrainedTokenizer,
        max_encoder_length: int = 512,
        max_decoder_length: int = 128,
    ):
        self.data = data
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.max_encoder_length = max_encoder_length
        self.max_decoder_length = max_decoder_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        source: str = item["source"]
        target: str = item["target"]

        # Tokenize source (input cho encoder mmBERT)
        enc = self.encoder_tokenizer(
            source,
            max_length=self.max_encoder_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Tokenize target (ground truth cho decoder mT5)
        with self.decoder_tokenizer.as_target_tokenizer():
            dec = self.decoder_tokenizer(
                target,
                max_length=self.max_decoder_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

        # LM labels: clone target ids, đổi padding_id thành -100
        lm_labels = dec["input_ids"].squeeze(0).clone()
        lm_labels[lm_labels == self.decoder_tokenizer.pad_token_id] = -100

        # Decoder input: shift right (thêm decoder_start_token vào đầu)
        # mT5 dùng pad_token_id làm decoder_start_token
        decoder_start_token_id = self.decoder_tokenizer.pad_token_id
        decoder_input_ids = torch.cat([
            torch.tensor([decoder_start_token_id]),
            dec["input_ids"].squeeze(0)[:-1],  # bỏ token cuối
        ])

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": dec["attention_mask"].squeeze(0),
            "lm_labels": lm_labels,
        }


class MultilingualMultiTaskDataset(Dataset):
    """
    Dataset kết hợp NER + Seq2Seq.
    Mỗi sample có thể có cả NER labels và target text.
    """

    def __init__(
        self,
        data: List[Dict],
        encoder_tokenizer: PreTrainedTokenizer,
        decoder_tokenizer: PreTrainedTokenizer,
        label2id: Dict[str, int],
        max_encoder_length: int = 512,
        max_decoder_length: int = 128,
        task_mode: str = "multitask",  # "ner", "seq2seq", "multitask"
    ):
        self.data = data
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.label2id = label2id
        self.max_encoder_length = max_encoder_length
        self.max_decoder_length = max_decoder_length
        self.task_mode = task_mode

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        result = {}

        # ----------------------------------------------------------------
        # Xác định source text
        # ----------------------------------------------------------------
        if "source" in item:
            source = item["source"]
        elif "text" in item:
            source = item["text"]
        elif "tokens" in item:
            source = " ".join(item["tokens"])
        else:
            raise ValueError(f"Sample {idx} không có 'source', 'text', hoặc 'tokens'")

        # ----------------------------------------------------------------
        # Tokenize encoder (word-level nếu có tokens, text nếu không)
        # ----------------------------------------------------------------
        if "tokens" in item and self.task_mode in ("ner", "multitask"):
            # Word-level tokenization để align NER labels
            tokens = item["tokens"]
            encoding = self.encoder_tokenizer(
                tokens,
                is_split_into_words=True,
                max_length=self.max_encoder_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            word_ids = encoding.word_ids(batch_index=0)
        else:
            encoding = self.encoder_tokenizer(
                source,
                max_length=self.max_encoder_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            word_ids = None

        result["input_ids"] = encoding["input_ids"].squeeze(0)
        result["attention_mask"] = encoding["attention_mask"].squeeze(0)

        # ----------------------------------------------------------------
        # NER labels
        # ----------------------------------------------------------------
        if self.task_mode in ("ner", "multitask") and "ner_tags" in item:
            ner_tags = item["ner_tags"]
            word_labels = [self.label2id.get(tag, 0) for tag in ner_tags]

            if word_ids is not None:
                # Align với subwords
                aligned_labels = []
                prev_word_idx = None
                for word_idx in word_ids:
                    if word_idx is None:
                        aligned_labels.append(-100)
                    elif word_idx != prev_word_idx:
                        val = word_labels[word_idx] if word_idx < len(word_labels) else -100
                        aligned_labels.append(val)
                    else:
                        aligned_labels.append(-100)
                    prev_word_idx = word_idx

                while len(aligned_labels) < self.max_encoder_length:
                    aligned_labels.append(-100)
                aligned_labels = aligned_labels[:self.max_encoder_length]
            else:
                aligned_labels = [-100] * self.max_encoder_length

            result["ner_labels"] = torch.tensor(aligned_labels, dtype=torch.long)

        # ----------------------------------------------------------------
        # Seq2Seq labels
        # ----------------------------------------------------------------
        if self.task_mode in ("seq2seq", "multitask") and "target" in item:
            target = item["target"]

            with self.decoder_tokenizer.as_target_tokenizer():
                dec = self.decoder_tokenizer(
                    target,
                    max_length=self.max_decoder_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )

            lm_labels = dec["input_ids"].squeeze(0).clone()
            lm_labels[lm_labels == self.decoder_tokenizer.pad_token_id] = -100

            decoder_start = self.decoder_tokenizer.pad_token_id
            decoder_input_ids = torch.cat([
                torch.tensor([decoder_start]),
                dec["input_ids"].squeeze(0)[:-1],
            ])

            result["decoder_input_ids"] = decoder_input_ids
            result["decoder_attention_mask"] = dec["attention_mask"].squeeze(0)
            result["lm_labels"] = lm_labels

        return result


# ================================================================
# UTILITY FUNCTIONS
# ================================================================

def load_json_data(file_path: str) -> List[Dict]:
    """Load data từ JSON file (list of dicts)."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[Data] Loaded {len(data)} samples từ {file_path}")
    return data


def create_sample_data(output_dir: str = "./data"):
    """
    Tạo sample data để test.
    Gọi hàm này để tạo file data mẫu nếu chưa có.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Sample NER + Seq2Seq data
    samples = [
        {
            "tokens": ["Barack", "Obama", "was", "born", "in", "Hawaii"],
            "ner_tags": ["B-PER", "I-PER", "O", "O", "O", "B-LOC"],
            "source": "Barack Obama was born in Hawaii",
            "target": "Barack Obama sinh ra ở Hawaii"
        },
        {
            "tokens": ["Apple", "is", "based", "in", "Cupertino"],
            "ner_tags": ["B-ORG", "O", "O", "O", "B-LOC"],
            "source": "Apple is based in Cupertino",
            "target": "Apple có trụ sở tại Cupertino"
        },
        {
            "tokens": ["The", "Eiffel", "Tower", "is", "in", "Paris"],
            "ner_tags": ["O", "B-LOC", "I-LOC", "O", "O", "B-LOC"],
            "source": "The Eiffel Tower is in Paris",
            "target": "Tháp Eiffel nằm ở Paris"
        },
    ]

    # Tạo train/val/test split
    n = len(samples)
    train_data = samples[:int(0.8*n)] if n > 2 else samples
    val_data = samples[int(0.8*n):int(0.9*n)] if n > 2 else samples[:1]
    test_data = samples[int(0.9*n):] if n > 2 else samples[:1]

    for split, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        path = os.path.join(output_dir, f"{split}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[Data] Đã tạo sample data tại {output_dir}/")


def get_dataloaders(
    config: Config,
    encoder_tokenizer: PreTrainedTokenizer,
    decoder_tokenizer: PreTrainedTokenizer,
    task_mode: str = "multitask",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Tạo DataLoaders cho train/val/test.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    mc = config.model
    tc = config.training
    dc = config.data

    # Load data
    train_data = load_json_data(dc.train_file)
    val_data = load_json_data(dc.val_file)
    test_data = load_json_data(dc.test_file)

    # Tạo datasets
    dataset_kwargs = dict(
        encoder_tokenizer=encoder_tokenizer,
        decoder_tokenizer=decoder_tokenizer,
        label2id=mc.ner_label2id,
        max_encoder_length=mc.max_encoder_length,
        max_decoder_length=mc.max_decoder_length,
        task_mode=task_mode,
    )

    train_dataset = MultilingualMultiTaskDataset(train_data, **dataset_kwargs)
    val_dataset = MultilingualMultiTaskDataset(val_data, **dataset_kwargs)
    test_dataset = MultilingualMultiTaskDataset(test_data, **dataset_kwargs)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=tc.batch_size,
        shuffle=True,
        num_workers=tc.dataloader_num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=tc.batch_size * 2,  # val không cần gradient, batch lớn hơn
        shuffle=False,
        num_workers=tc.dataloader_num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=tc.batch_size * 2,
        shuffle=False,
        num_workers=tc.dataloader_num_workers,
        pin_memory=True,
    )

    print(f"[Data] Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    return train_loader, val_loader, test_loader