"""
inference.py
============
Inference pipeline cho Multilingual NLP Model.

Hỗ trợ:
  - NER prediction cho đơn lẻ hoặc batch
  - Text generation (seq2seq)
  - Kết hợp cả hai trong cùng một call

Ví dụ sử dụng nhanh:

    pipeline = InferencePipeline.from_checkpoint("./checkpoints/checkpoint_best_xxx")
    
    # NER
    results = pipeline.predict_ner("Barack Obama was born in Hawaii")
    # [{"token": "Barack", "label": "B-PER"}, {"token": "Obama", "label": "I-PER"}, ...]
    
    # Generation
    text = pipeline.generate("Hello, how are you?", max_new_tokens=50)
    # "Xin chào, bạn khỏe không?"
"""

import torch
import json
import os
from typing import List, Dict, Optional, Union, Tuple
from transformers import AutoTokenizer

from config import Config, DEFAULT_CONFIG
from model import MultilingualNLPModel


class InferencePipeline:
    """
    Pipeline inference tiện dụng.
    Tự động xử lý tokenization và decode output.
    """

    def __init__(
        self,
        model: MultilingualNLPModel,
        encoder_tokenizer,
        decoder_tokenizer,
        config: Config = DEFAULT_CONFIG,
        device: Optional[str] = None,
    ):
        self.model = model
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.config = config
        mc = config.model

        # Device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Label maps
        self.id2label = mc.ner_id2label
        self.label2id = mc.ner_label2id

        print(f"[Inference] Pipeline sẵn sàng trên device: {self.device}")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str,
        config: Optional[Config] = None,
        device: Optional[str] = None,
    ) -> "InferencePipeline":
        """
        Load pipeline từ checkpoint đã save.

        Args:
            checkpoint_dir: Thư mục chứa checkpoint (có model.pt, config.json...)
            config:         Config object (nếu None đọc từ checkpoint)
            device:         "cuda", "cpu", hay "mps"

        Returns:
            InferencePipeline sẵn sàng dùng
        """
        # Load model
        model = MultilingualNLPModel.load(checkpoint_dir, config)

        # Load config từ checkpoint nếu cần
        if config is None:
            config_path = os.path.join(checkpoint_dir, "config.json")
            if os.path.exists(config_path):
                # Đã được load trong MultilingualNLPModel.load()
                config = model.config
            else:
                config = DEFAULT_CONFIG

        # Load tokenizers
        mc = config.model
        print(f"[Inference] Loading encoder tokenizer từ: {mc.encoder_name}")
        encoder_tokenizer = AutoTokenizer.from_pretrained(mc.encoder_name)

        print(f"[Inference] Loading decoder tokenizer từ: {mc.decoder_name}")
        decoder_tokenizer = AutoTokenizer.from_pretrained(mc.decoder_name)

        return cls(model, encoder_tokenizer, decoder_tokenizer, config, device)

    def _encode_text(
        self,
        text: Union[str, List[str]],
        return_word_ids: bool = False,
    ) -> Dict:
        """
        Tokenize text và trả về tensors đã move sang device.

        Args:
            text:           String hoặc list of strings
            return_word_ids: Có trả về word_ids không (dùng cho NER)

        Returns:
            encoding dict với input_ids, attention_mask trên device
        """
        is_single = isinstance(text, str)
        texts = [text] if is_single else text

        encoding = self.encoder_tokenizer(
            texts,
            max_length=self.config.model.max_encoder_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        result = {
            "input_ids": encoding["input_ids"].to(self.device),
            "attention_mask": encoding["attention_mask"].to(self.device),
            "is_single": is_single,
        }

        if return_word_ids:
            result["word_ids"] = [encoding.word_ids(i) for i in range(len(texts))]

        return result

    @torch.no_grad()
    def predict_ner(
        self,
        text: Union[str, List[str]],
        return_entities_only: bool = False,
    ) -> Union[List[Dict], List[List[Dict]]]:
        """
        Dự đoán NER cho text.

        Args:
            text:                String hoặc list of strings
            return_entities_only: Nếu True, chỉ trả về entities (không phải "O")

        Returns:
            Nếu input là string: List[Dict] mỗi dict là {"token": str, "label": str, "score": float}
            Nếu input là list:   List[List[Dict]]

        Ví dụ:
            pipeline.predict_ner("Barack Obama was born in Hawaii")
            → [
                {"token": "Barack", "label": "B-PER", "score": 0.99},
                {"token": "Obama",  "label": "I-PER", "score": 0.98},
                {"token": "was",    "label": "O",     "score": 0.99},
                ...
              ]
        """
        is_single = isinstance(text, str)
        texts = [text] if is_single else text

        # Tokenize
        encoding = self.encoder_tokenizer(
            texts,
            max_length=self.config.model.max_encoder_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # Forward (chỉ cần encoder + NER head)
        hidden_states = self.model.encode(input_ids, attention_mask)
        ner_hidden = self.model.ner_dropout(hidden_states)
        ner_logits = self.model.ner_classifier(ner_hidden)  # [batch, seq_len, num_labels]

        # Softmax để lấy xác suất
        ner_probs = torch.softmax(ner_logits, dim=-1)   # [batch, seq_len, num_labels]
        ner_preds = ner_probs.argmax(dim=-1)             # [batch, seq_len]

        # Decode kết quả
        all_results = []

        for i, text_i in enumerate(texts):
            word_ids = encoding.word_ids(batch_index=i)
            tokens = self.encoder_tokenizer.convert_ids_to_tokens(input_ids[i])

            results_i = []
            seen_words = set()

            for j, (word_idx, token) in enumerate(zip(word_ids, tokens)):
                if word_idx is None:
                    continue  # Bỏ special tokens
                if word_idx in seen_words:
                    continue  # Chỉ lấy subword đầu tiên của mỗi word
                seen_words.add(word_idx)

                pred_id = ner_preds[i][j].item()
                label = self.id2label.get(pred_id, "O")
                score = ner_probs[i][j][pred_id].item()

                if return_entities_only and label == "O":
                    continue

                # Lấy token gốc (trước khi tokenize thành subwords)
                # Cách đơn giản: dùng token của subword đầu
                clean_token = token.replace("▁", "").replace("##", "")

                results_i.append({
                    "token": clean_token,
                    "label": label,
                    "score": round(score, 4),
                    "word_idx": word_idx,
                })

            all_results.append(results_i)

        return all_results[0] if is_single else all_results

    @torch.no_grad()
    def extract_entities(
        self,
        text: Union[str, List[str]],
    ) -> Union[List[Dict], List[List[Dict]]]:
        """
        Trích xuất entities đã gộp (gộp B- và I- tokens lại).

        Returns:
            List[Dict] với keys: "text", "label", "score", "start", "end"

        Ví dụ:
            pipeline.extract_entities("Barack Obama was born in Hawaii")
            → [
                {"text": "Barack Obama", "label": "PER", "score": 0.985},
                {"text": "Hawaii",       "label": "LOC", "score": 0.992},
              ]
        """
        is_single = isinstance(text, str)
        texts = [text] if is_single else text

        all_entities = []
        for text_i in texts:
            token_results = self.predict_ner(text_i)
            entities = self._merge_bio_entities(token_results)
            all_entities.append(entities)

        return all_entities[0] if is_single else all_entities

    def _merge_bio_entities(self, token_results: List[Dict]) -> List[Dict]:
        """Gộp các B- và I- tokens liên tiếp thành 1 entity."""
        entities = []
        current_entity = None

        for item in token_results:
            label = item["label"]
            token = item["token"]
            score = item["score"]

            if label.startswith("B-"):
                # Bắt đầu entity mới
                if current_entity is not None:
                    entities.append(current_entity)
                entity_type = label[2:]  # Bỏ "B-"
                current_entity = {
                    "text": token,
                    "label": entity_type,
                    "scores": [score],
                }

            elif label.startswith("I-") and current_entity is not None:
                entity_type = label[2:]  # Bỏ "I-"
                if entity_type == current_entity["label"]:
                    # Tiếp tục entity hiện tại
                    current_entity["text"] += " " + token
                    current_entity["scores"].append(score)
                else:
                    # Type không khớp, kết thúc entity cũ
                    entities.append(current_entity)
                    current_entity = None
            else:
                # "O" - kết thúc entity nếu đang có
                if current_entity is not None:
                    entities.append(current_entity)
                    current_entity = None

        if current_entity is not None:
            entities.append(current_entity)

        # Tính score trung bình
        for entity in entities:
            entity["score"] = round(sum(entity.pop("scores")) / len(entity.get("scores", [1])), 4)

        return entities

    @torch.no_grad()
    def generate(
        self,
        text: Union[str, List[str]],
        max_new_tokens: int = 128,
        num_beams: int = 4,
        num_return_sequences: int = 1,
        **generation_kwargs,
    ) -> Union[str, List[str]]:
        """
        Sinh text từ input (dịch thuật, tóm tắt...).

        Args:
            text:                 Input text (string hoặc list)
            max_new_tokens:       Số token tối đa sinh ra
            num_beams:            Số beams cho beam search
            num_return_sequences: Số sequences trả về mỗi input

        Returns:
            String hoặc List[str] tùy input

        Ví dụ:
            pipeline.generate("Hello, how are you?")
            → "Xin chào, bạn khỏe không?"
        """
        is_single = isinstance(text, str)
        texts = [text] if is_single else text

        # Tokenize input
        encoding = self.encoder_tokenizer(
            texts,
            max_length=self.config.model.max_encoder_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # Generate
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            **generation_kwargs,
        )

        # Decode
        decoded = self.decoder_tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )

        if is_single:
            return decoded[0] if num_return_sequences == 1 else decoded
        return decoded

    @torch.no_grad()
    def analyze(self, text: str) -> Dict:
        """
        Phân tích toàn diện: NER + Generation cùng lúc.

        Returns:
            Dict với keys: "entities", "generated_text"
        """
        entities = self.extract_entities(text)
        generated = self.generate(text)

        return {
            "input_text": text,
            "entities": entities,
            "generated_text": generated,
        }