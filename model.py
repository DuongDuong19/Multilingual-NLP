"""
model.py
Pipeline: raw text → Tokenizer → mmBERT-small → NER → Projector → mT5-small decoder → text
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Dùng Auto classes để tránh import chain dài gây lỗi torchvision
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,   # load mT5 qua Auto thay vì MT5ForConditionalGeneration
)
from transformers import T5Tokenizer   # MT5Tokenizer kế thừa T5Tokenizer, tránh import MT5Tokenizer trực tiếp


class MTT(nn.Module):
    """
    Pipeline hoàn chỉnh:
        raw src text
          ──► enc_tokenizer             (AutoTokenizer / mmBERT-small)
          ──► mmBERT-small encoder      → enc_hidden    (B, src_len, enc_dim=512)
          ──► NERClassifier             → ner_logits    (B, src_len, 10)
          ──► nerEmbed(argmax)          → ner_emb       (B, src_len, 512)
          ──► enc_hidden + ner_emb
          ──► Projector (Linear)        → projected     (B, src_len, dec_dim=512)
          ──► mT5-small decoder stack   → dec_hidden    (B, tgt_len, 512)
          ──► LM head                   → logits        (B, tgt_len, vocab_size)
          ──► decoded text
    """

    NER_LAMBDA  = 0.3
    MAX_SRC_LEN = 128
    MAX_TGT_LEN = 128

    NER_VOCAB = {
        'O': 0, 'PERSON': 1, 'ORG': 2, 'LOC': 3, 'GPE': 4,
        'DATE': 5, 'MONEY': 6, 'PERCENT': 7, 'TIME': 8, 'QUANTITY': 9,
    }

    def __init__(self):
        super().__init__()

        # ── 1. Tokenizers ────────────────────────────────────────────────────
        self.enc_tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-small")
        self.dec_tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")

        # ── 2. mmBERT-small Encoder ──────────────────────────────────────────
        self.encoder = AutoModel.from_pretrained("jhu-clsp/mmBERT-small")
        enc_dim = self.encoder.config.hidden_size   # 512

        # ── 3. NER head ──────────────────────────────────────────────────────
        self.nerEmbed = nn.Embedding(
            num_embeddings=len(self.NER_VOCAB),
            embedding_dim=enc_dim,
        )
        self.nerClassifier = nn.Linear(enc_dim, len(self.NER_VOCAB))

        # ── 4. mT5-small – lấy decoder stack + lm_head, bỏ encoder ──────────
        _mt5    = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")
        dec_dim = _mt5.config.hidden_size   # 512

        self.decoder = _mt5.decoder         # MT5Stack
        self.lm_head = _mt5.lm_head         # Linear(512, vocab_size, bias=False)
        del _mt5                             # giải phóng encoder của mT5

        # ── 5. Projector: enc_dim → dec_dim ──────────────────────────────────
        self.projector = nn.Linear(enc_dim, dec_dim)

        # Toàn bộ params đều train
        for p in self.parameters():
            p.requires_grad = True

    # =========================================================================
    # INTERNAL: TOKENIZE
    # =========================================================================

    def _tokenize_src(self, texts: list[str]):
        """Tokenize raw source text bằng mmBERT tokenizer."""
        device = next(self.encoder.parameters()).device
        enc = self.enc_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.MAX_SRC_LEN,
            return_tensors="pt",
        )
        # ModernBERT (jhu-clsp/mmBERT-small) không có token_type_ids
        return {
            "input_ids":      enc["input_ids"].to(device),
            "attention_mask": enc["attention_mask"].to(device),
        }

    def _tokenize_tgt(self, texts: list[str]):
        """
        Tokenize raw target text bằng mT5 tokenizer.
        Trả về:
          decoder_input_ids  – shifted right  (BOS = pad_token_id theo T5/mT5 convention)
          labels             – padding → -100
        """
        device = next(self.encoder.parameters()).device
        enc = self.dec_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.MAX_TGT_LEN,
            return_tensors="pt",
        )
        ids = enc["input_ids"].to(device)

        # Shift right
        bos = torch.full(
            (ids.size(0), 1),
            self.dec_tokenizer.pad_token_id,
            dtype=torch.long, device=device,
        )
        decoder_input_ids = torch.cat([bos, ids[:, :-1]], dim=1)

        # Labels: padding → -100
        labels = ids.clone()
        labels[labels == self.dec_tokenizer.pad_token_id] = -100

        return decoder_input_ids, labels

    # =========================================================================
    # FORWARD
    # =========================================================================

    def forward(
        self,
        src_texts: list[str],   # raw source strings
        tgt_texts: list[str],   # raw target strings
        ner_labels=None,        # (B, src_len) int64 tensor, -100 ở padding, optional
    ):
        # Step 1: Tokenize
        src_enc                   = self._tokenize_src(src_texts)
        decoder_input_ids, labels = self._tokenize_tgt(tgt_texts)

        # Step 2: mmBERT-small encode
        enc_hidden = self.encoder(
            input_ids=src_enc["input_ids"],
            attention_mask=src_enc["attention_mask"],
        ).last_hidden_state                                    # (B, src_len, 512)

        # Step 3: NER classify + embed
        ner_logits = self.nerClassifier(enc_hidden)            # (B, src_len, 10)
        ner_emb    = self.nerEmbed(ner_logits.argmax(-1))      # (B, src_len, 512)

        # Step 4: Inject NER + Project
        projected = self.projector(enc_hidden + ner_emb)       # (B, src_len, 512)

        # Step 5: mT5 decoder
        dec_hidden = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=projected,
            encoder_attention_mask=src_enc["attention_mask"],
        ).last_hidden_state                                    # (B, tgt_len, 512)

        # Step 6: LM head
        logits = self.lm_head(dec_hidden)                      # (B, tgt_len, vocab)

        # Step 7: Generation loss
        gen_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )

        # Step 8: NER loss (nếu có)
        ner_loss = torch.tensor(0.0, device=enc_hidden.device)
        if ner_labels is not None:
            ner_loss = F.cross_entropy(
                ner_logits.reshape(-1, len(self.NER_VOCAB)),
                ner_labels.reshape(-1),
                ignore_index=-100,
            )

        total_loss = gen_loss + self.NER_LAMBDA * ner_loss
        return total_loss, gen_loss.detach(), ner_loss.detach()

    # =========================================================================
    # GENERATE (greedy decode)
    # =========================================================================

    @torch.no_grad()
    def generate(self, src_texts: list[str], max_new_tokens: int = 128):
        """Input: raw string list → Output: decoded string list."""
        device = next(self.encoder.parameters()).device
        B      = len(src_texts)

        # Tokenize + Encode
        src_enc    = self._tokenize_src(src_texts)
        enc_hidden = self.encoder(
            input_ids=src_enc["input_ids"],
            attention_mask=src_enc["attention_mask"],
        ).last_hidden_state

        # NER + Project
        ner_emb   = self.nerEmbed(self.nerClassifier(enc_hidden).argmax(-1))
        projected = self.projector(enc_hidden + ner_emb)

        # Greedy decode với KV cache
        dec_ids = torch.full(
            (B, 1), self.dec_tokenizer.pad_token_id,
            dtype=torch.long, device=device,
        )
        past_kv = None

        for _ in range(max_new_tokens):
            cur_input = dec_ids if past_kv is None else dec_ids[:, -1:]
            dec_out   = self.decoder(
                input_ids=cur_input,
                encoder_hidden_states=projected,
                encoder_attention_mask=src_enc["attention_mask"],
                past_key_values=past_kv,
                use_cache=True,
            )
            past_kv  = dec_out.past_key_values
            next_tok = self.lm_head(dec_out.last_hidden_state[:, -1]).argmax(-1, keepdim=True)
            dec_ids  = torch.cat([dec_ids, next_tok], dim=1)
            if (next_tok == self.dec_tokenizer.eos_token_id).all():
                break

        return self.dec_tokenizer.batch_decode(dec_ids[:, 1:], skip_special_tokens=True)

    # =========================================================================
    # UTILS
    # =========================================================================

    def paramsCalc(self):
        parts = {
            "encoder"      : self.encoder,
            "nerEmbed"     : self.nerEmbed,
            "nerClassifier": self.nerClassifier,
            "projector"    : self.projector,
            "decoder"      : self.decoder,
            "lm_head"      : self.lm_head,
        }
        total = 0
        for name, module in parts.items():
            n = sum(p.numel() for p in module.parameters())
            total += n
            print(f"  {name:<16}: {n:>12,}")
        print(f"  {'TOTAL':<16}: {total:>12,}")

    def enable_gradient_checkpointing(self):
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()
        if hasattr(self.decoder, "gradient_checkpointing_enable"):
            self.decoder.gradient_checkpointing_enable()


# ─── Quick test khi chạy trực tiếp ───────────────────────────────────────────
if __name__ == "__main__":
    print("[TEST] Khởi tạo MTT model…")
    model = MTT()
    print("[TEST] Param count:")
    model.paramsCalc()

    print("\n[TEST] Forward pass với dummy data…")
    loss, gen, ner = model(
        src_texts=["The president of France visited Vietnam."],
        tgt_texts=["Tổng thống Pháp đã thăm Việt Nam."],
    )
    print(f"  total_loss={loss.item():.4f}  gen={gen.item():.4f}  ner={ner.item():.4f}")

    print("\n[TEST] Generate…")
    out = model.generate(["Hello world, this is a test."])
    print(f"  output: {out}")

    print("\n[OK] model.py chạy thành công!")