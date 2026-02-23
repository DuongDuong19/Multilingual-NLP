# Multilingual-NLP

##Install
```bibtex
pip install transformers protobuf seqeval accelerate sentencepiece tiktoken
pip install "datasets<3.0.0"
```

##Run
```bibtex
1. python models.py
2. python train.py  (Full train (3 epochs, auto-save 30 phút))
3. # Only train NER head
   python train.py --freeze_encoder --epochs 5
4. # Custom hyperparams
   python train.py --epochs 5 --batch_size 32 --lr_mmbert 1e-5 --lr_mt5 2e-5
5. # Resume sau crash
   python train.py --resume_mmbert ./checkpoints/mmBERT-small/mmBERT-small_best.pt
