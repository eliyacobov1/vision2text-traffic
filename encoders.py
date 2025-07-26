# Vision and Text encoder modules
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, DistilBertConfig, DistilBertModel
import timm


class SimpleTokenizer:
    """Fallback whitespace tokenizer if HF models are unavailable."""

    def __init__(self, max_length: int = 32):
        self.max_length = max_length
        self.vocab = {"[PAD]": 0, "[UNK]": 1}

    def __call__(self, text, return_tensors=None, padding=True, truncation=True, max_length=None):
        max_length = max_length or self.max_length
        tokens = text.lower().split()
        ids = []
        for t in tokens:
            if t not in self.vocab:
                self.vocab[t] = len(self.vocab)
            ids.append(self.vocab[t])
        ids = ids[:max_length]
        attn = [1] * len(ids)
        while len(ids) < max_length:
            ids.append(0)
            attn.append(0)
        out = {"input_ids": torch.tensor([ids]), "attention_mask": torch.tensor([attn])}
        return out


class VisionEncoder(nn.Module):
    def __init__(self, name: str, freeze: bool = True):
        super().__init__()
        model = timm.create_model(name, pretrained=True)
        model.reset_classifier(0)
        self.model = model
        self.out_dim = model.num_features
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model.forward_features(images)


class TextEncoder(nn.Module):
    def __init__(self, name: str, freeze: bool = True, offline: bool = True):
        super().__init__()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=offline)
            self.model = AutoModel.from_pretrained(name, local_files_only=offline)
        except Exception:
            self.tokenizer = SimpleTokenizer()
            self.model = DistilBertModel(DistilBertConfig())
        self.out_dim = self.model.config.hidden_size
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state
