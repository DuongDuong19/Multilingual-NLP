"""
File: src/model.py
Mô tả: Định nghĩa MultilingualNERTranslationModel
"""

import torch
import torch.nn as nn
import spacy
from transformers import (
    BertModel, 
    BertTokenizer,
    MT5ForConditionalGeneration,
    AutoTokenizer,          # ← Thay MT5Tokenizer bằng AutoTokenizer
)


class MultilingualNERTranslationModel(nn.Module):
    """
    Model dịch thuật kết hợp NER với kiến trúc:
    - mBERT encoder
    - spaCy NER extraction
    - mT5 decoder
    """
    
    def __init__(
        self, 
        mbert_model_name='bert-base-multilingual-cased',
        mt5_model_name='google/mt5-small',
        max_length=128
    ):
        super().__init__()
        
        self.max_length = max_length
        
        # ===== ENCODER: mBERT =====
        self.encoder = BertModel.from_pretrained(mbert_model_name)
        self.encoder_tokenizer = BertTokenizer.from_pretrained(mbert_model_name)
        
        # ===== DECODER: mT5 =====
        self.decoder = MT5ForConditionalGeneration.from_pretrained(mt5_model_name)
        self.decoder_tokenizer = AutoTokenizer.from_pretrained(mt5_model_name)
        
        # ===== PROJECTION LAYER =====
        self.encoder_to_decoder = nn.Linear(
            self.encoder.config.hidden_size,  # 768 cho bert-base
            self.decoder.config.d_model       # 512 cho mt5-small
        )
        
        # ===== NER EMBEDDING LAYER =====
        self.ner_vocab = {
            'O': 0, 'PERSON': 1, 'ORG': 2, 'LOC': 3, 'GPE': 4,
            'DATE': 5, 'MONEY': 6, 'PERCENT': 7, 'TIME': 8, 'QUANTITY': 9,
        }
        
        self.ner_embedding = nn.Embedding(
            num_embeddings=len(self.ner_vocab),
            embedding_dim=self.encoder.config.hidden_size
        )
        
        # Load spaCy NER model
        self.nlp = spacy.load("xx_ent_wiki_sm")
    
    def extract_ner_tags(self, text):
        """Trích xuất NER tags từ văn bản sử dụng spaCy"""
        doc = self.nlp(text)
        bert_tokens = self.encoder_tokenizer.tokenize(text)
        ner_tags = [self.ner_vocab['O']] * len(bert_tokens)
        
        # Map entities từ spaCy sang BERT tokens
        char_to_token = {}
        current_pos = 0
        
        for idx, token in enumerate(bert_tokens):
            token_text = token.replace('##', '')
            char_to_token[current_pos] = idx
            current_pos += len(token_text)
        
        # Gán NER tags
        for ent in doc.ents:
            entity_type = ent.label_
            if entity_type in self.ner_vocab:
                tag_id = self.ner_vocab[entity_type]
                start_char = ent.start_char
                end_char = ent.end_char
                
                for char_pos, token_idx in char_to_token.items():
                    if start_char <= char_pos < end_char:
                        ner_tags[token_idx] = tag_id
        
        return ner_tags
    
    def forward(self, source_texts, target_texts=None):
        """Forward pass của model"""
        batch_size = len(source_texts)
        device = self.encoder.device
        
        # Tokenize source
        encoder_inputs = self.encoder_tokenizer(
            source_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(device)
        
        # Extract NER tags
        all_ner_tags = []
        for text in source_texts:
            ner_tags = self.extract_ner_tags(text)
            all_ner_tags.append(ner_tags)
        
        # Pad NER tags
        max_seq_len = encoder_inputs['input_ids'].size(1)
        padded_ner_tags = []
        
        for ner_tags in all_ner_tags:
            if len(ner_tags) > max_seq_len:
                ner_tags = ner_tags[:max_seq_len]
            else:
                ner_tags = ner_tags + [self.ner_vocab['O']] * (max_seq_len - len(ner_tags))
            padded_ner_tags.append(ner_tags)
        
        ner_tags_tensor = torch.tensor(padded_ner_tags).to(device)
        
        # Encode với mBERT
        encoder_outputs = self.encoder(**encoder_inputs)
        encoder_hidden_states = encoder_outputs.last_hidden_state
        
        # Add NER embeddings
        ner_embeds = self.ner_embedding(ner_tags_tensor)
        encoder_hidden_states = encoder_hidden_states + ner_embeds
        
        # Project sang decoder dimension
        encoder_hidden_states = self.encoder_to_decoder(encoder_hidden_states)
        
        # Decoder
        if target_texts is not None:
            # Training mode
            decoder_inputs = self.decoder_tokenizer(
                target_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.max_length
            ).to(device)
            
            labels = decoder_inputs['input_ids'].clone()
            labels[labels == self.decoder_tokenizer.pad_token_id] = -100
            
            outputs = self.decoder(
                encoder_outputs=[encoder_hidden_states],
                attention_mask=encoder_inputs['attention_mask'],
                labels=labels
            )
            
            return outputs
        else:
            # Inference mode
            return encoder_hidden_states, encoder_inputs['attention_mask']
    
    def generate(self, source_texts, max_length=128, num_beams=4):
        """Sinh văn bản dịch"""
        if isinstance(source_texts, str):
            source_texts = [source_texts]
            return_single = True
        else:
            return_single = False
        
        self.eval()
        with torch.no_grad():
            encoder_hidden_states, attention_mask = self.forward(source_texts)
            
            generated_ids = self.decoder.generate(
                encoder_outputs=[encoder_hidden_states],
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
            
            generated_texts = self.decoder_tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )
        
        if return_single:
            return generated_texts[0]
        return generated_texts
    
    def save_model(self, save_dir):
        """Lưu model"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        torch.save(self.state_dict(), os.path.join(save_dir, 'model.pt'))
        self.encoder_tokenizer.save_pretrained(os.path.join(save_dir, 'encoder_tokenizer'))
        self.decoder_tokenizer.save_pretrained(os.path.join(save_dir, 'decoder_tokenizer'))
        
        import json
        config = {
            'max_length': self.max_length,
            'ner_vocab': self.ner_vocab,
        }
        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            json.dump(config, f)
    
    def load_model(self, save_dir):
        """Load model"""
        import os
        import json
        
        with open(os.path.join(save_dir, 'config.json'), 'r') as f:
            config = json.load(f)
        
        self.load_state_dict(torch.load(os.path.join(save_dir, 'model.pt')))