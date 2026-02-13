"""
File: src/trainer.py
Mô tả: Training và evaluation logic
"""

import torch
from tqdm.auto import tqdm
from torch.cuda.amp import autocast, GradScaler
from sacrebleu.metrics import BLEU


class NERTranslationTrainer:
    """Trainer class cho NER Translation model"""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        use_amp=True
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.use_amp = use_amp
        self.scaler = GradScaler() if use_amp else None
    
    def train_epoch(self):
        """Train 1 epoch"""
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            source_texts = batch['source_texts']
            target_texts = batch['target_texts']
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    outputs = self.model(source_texts, target_texts)
                    loss = outputs.loss
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(source_texts, target_texts)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            self.scheduler.step()
            total_loss += loss.item()
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/(batch_idx+1):.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
        
        return total_loss / len(self.train_loader)
    
    def evaluate(self):
        """Evaluate trên validation set"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                source_texts = batch['source_texts']
                target_texts = batch['target_texts']
                
                if self.use_amp:
                    with autocast():
                        outputs = self.model(source_texts, target_texts)
                        loss = outputs.loss
                else:
                    outputs = self.model(source_texts, target_texts)
                    loss = outputs.loss
                
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def compute_bleu(self, test_loader, num_samples=100):
        """Tính BLEU score"""
        self.model.eval()
        bleu = BLEU()
        
        predictions = []
        references = []
        
        count = 0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Computing BLEU"):
                source_texts = batch['source_texts']
                target_texts = batch['target_texts']
                
                translations = self.model.generate(
                    source_texts, 
                    max_length=128, 
                    num_beams=4
                )
                
                predictions.extend(translations)
                references.extend([[t] for t in target_texts])
                
                count += len(source_texts)
                if count >= num_samples:
                    break
        
        score = bleu.corpus_score(predictions, references)
        return score.score, predictions, references