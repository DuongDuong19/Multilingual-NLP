import torch.nn as nn
from transformers import (
    AutoModel,
    AutoTokenizer,
    MT5ForConditionalGeneration,
)

class MTT(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = AutoModel.from_pretrained("jhu-clsp/mmBERT-small")

        self.nerVocab = {
            'O': 0, 'PERSON': 1, 'ORG': 2, 'LOC': 3, 'GPE': 4,
            'DATE': 5, 'MONEY': 6, 'PERCENT': 7, 'TIME': 8, 'QUANTITY': 9
        }
        self.nerEmbed = nn.Embedding(
            num_embeddings = len(self.nerVocab),
            embedding_dim = self.encoder.config.hidden_size
        )
        self.nerClassifier = nn.Linear(
            self.encoder.config.hidden_size,
            len(self.nerVocab)
        )
        
        self.decoder = MT5ForConditionalGeneration.from_pretrained("google/mt5-small").decoder

        for param in self.encoder.parameters():
            param.requires_grad = True

        for param in self.decoder.parameters():
            param.requires_grad = True
 
        self.projector = nn.Linear(self.encoder.config.hidden_size, self.decoder.config.hidden_size)

    def paramsCalc(self):
        encoderPara = sum(p.numel() for p in self.encoder.parameters())
        embedPara = sum(p.numel() for p in self.nerEmbed.parameters())
        nerPara = sum(p.numel() for p in self.nerClassifier.parameters())
        proPara = sum(p.numel() for p in self.projector.parameters())
        decoderPara = sum(p.numel() for p in self.decoder.parameters())

        print("Total: ", encoderPara + embedPara + nerPara + proPara + decoderPara)
