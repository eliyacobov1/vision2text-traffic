import torch
import torch.nn as nn
from dataclasses import dataclass
from transformers import AutoModel, AutoTokenizer, DistilBertConfig, DistilBertModel


class SimpleTokenizer:
    """Fallback tokenizer using whitespace tokenization."""

    def __init__(self, max_length: int = 32):
        self.max_length = max_length
        self.vocab = {"[PAD]": 0, "[UNK]": 1}

    def __call__(self, text, return_tensors=None, padding=True, truncation=True, max_length=None):
        max_length = max_length or self.max_length
        tokens = text.lower().split()
        ids = []
        for tok in tokens:
            if tok not in self.vocab:
                self.vocab[tok] = len(self.vocab)
            ids.append(self.vocab[tok])
        ids = ids[:max_length]
        attn = [1] * len(ids)
        while len(ids) < max_length:
            ids.append(0)
            attn.append(0)
        res = {"input_ids": torch.tensor([ids]), "attention_mask": torch.tensor([attn])}
        return res
from torchvision import models

@dataclass
class VLTConfig:
    vision_model: str = 'resnet18'
    text_model: str = 'distilbert-base-uncased'
    hidden_dim: int = 256
    num_heads: int = 8

class VisionLanguageTransformer(nn.Module):
    """Vision-Language model for congestion classification with cross-attention."""

    def __init__(self, config: VLTConfig = VLTConfig()):
        super().__init__()
        self.config = config
        # Vision encoder
        if config.vision_model.startswith('resnet'):
            # Use no pretrained weights if download is unavailable
            vision = getattr(models, config.vision_model)(weights=None)
            vision_out = vision.fc.in_features
            vision.fc = nn.Identity()
        else:
            raise ValueError(f'Unsupported vision model {config.vision_model}')
        self.vision_encoder = vision
        # Text encoder
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(config.text_model, local_files_only=True)
            text_encoder = AutoModel.from_pretrained(config.text_model, local_files_only=True)
        except Exception:
            # Fallback to randomly initialized DistilBERT with a simple tokenizer
            self.tokenizer = SimpleTokenizer()
            text_encoder = DistilBertModel(DistilBertConfig())
        text_out = text_encoder.config.hidden_size
        self.text_encoder = text_encoder

        # Freeze encoders
        for p in self.vision_encoder.parameters():
            p.requires_grad = False
        for p in self.text_encoder.parameters():
            p.requires_grad = False

        # Projection layers
        self.img_proj = nn.Linear(vision_out, config.hidden_dim)
        self.txt_proj = nn.Linear(text_out, config.hidden_dim)
        
        self.cross_attn = nn.MultiheadAttention(config.hidden_dim, config.num_heads, batch_first=True)
        self.classifier = nn.Linear(config.hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, images, input_ids, attention_mask):
        # Encode modalities
        with torch.no_grad():
            img_feat = self.vision_encoder(images)  # (B, feat)
            text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            text_feat = text_outputs.last_hidden_state  # (B, L, D)
        img_feat = self.img_proj(img_feat).unsqueeze(1)  # (B,1,H)
        txt_feat = self.txt_proj(text_feat)             # (B,L,H)

        # Cross-attention from text to image
        fused, _ = self.cross_attn(txt_feat, img_feat, img_feat)
        cls = fused[:,0,:]  # use CLS token from text
        logits = self.classifier(cls)
        return self.sigmoid(logits).squeeze(-1)

    def encode_text(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        if hasattr(self.tokenizer, '__call__') and not isinstance(self.tokenizer, SimpleTokenizer):
            tokens = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
            return tokens['input_ids'], tokens['attention_mask']
        else:
            ids = []
            masks = []
            for t in texts:
                out = self.tokenizer(t)
                ids.append(out['input_ids'].squeeze(0))
                masks.append(out['attention_mask'].squeeze(0))
            return torch.stack(ids), torch.stack(masks)
