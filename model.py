import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, List

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


@dataclass
class VLTConfig:
    vision_model: str = "vit_base_patch16_224"
    text_model: str = "distilbert-base-uncased"
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 2
    contrastive: bool = False
    loss_type: str = "classification"  # classification | contrastive | hybrid


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


class CrossAttentionBlock(nn.Module):
    """Single cross-attention + feed-forward block."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.last_attn: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        q = self.norm1(x)
        attn_out, attn_w = self.attn(q, context, context, need_weights=True)
        self.last_attn = attn_w  # (B, heads, Q, K)
        x = x + attn_out
        ff_out = self.ff(self.norm2(x))
        return x + ff_out


class CrossModalFusion(nn.Module):
    def __init__(self, dim: int, num_heads: int, depth: int):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttentionBlock(dim, num_heads) for _ in range(depth)
        ])

    def forward(self, txt_tokens: torch.Tensor, img_tokens: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            txt_tokens = layer(txt_tokens, img_tokens)
        return txt_tokens

    def get_last_attention(self) -> torch.Tensor | None:
        if self.layers:
            return self.layers[-1].last_attn
        return None


class ClassificationHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, 1)

    def forward(self, cls: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc(cls)).squeeze(-1)


class ContrastiveHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.img_proj = nn.Linear(dim, dim)
        self.txt_proj = nn.Linear(dim, dim)

    def forward(self, img_tokens: torch.Tensor, txt_cls: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        img_feat = img_tokens.mean(dim=1)
        img_emb = F.normalize(self.img_proj(img_feat), dim=-1)
        txt_emb = F.normalize(self.txt_proj(txt_cls), dim=-1)
        return img_emb, txt_emb


class VisionLanguageTransformer(nn.Module):
    def __init__(self, config: VLTConfig = VLTConfig(), offline: bool = True):
        super().__init__()
        self.config = config
        self.vision = VisionEncoder(config.vision_model, freeze=True)
        self.text = TextEncoder(config.text_model, freeze=True, offline=offline)

        self.img_proj = nn.Linear(self.vision.out_dim, config.hidden_dim)
        self.txt_proj = nn.Linear(self.text.out_dim, config.hidden_dim)

        self.fusion = CrossModalFusion(config.hidden_dim, config.num_heads, config.num_layers)

        self.class_head = ClassificationHead(config.hidden_dim)
        self.contrastive_head = ContrastiveHead(config.hidden_dim) if config.contrastive else None

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        img_tokens = self.img_proj(self.vision(images))
        txt_tokens = self.txt_proj(self.text(input_ids, attention_mask))
        fused = self.fusion(txt_tokens, img_tokens)
        cls = fused[:, 0, :]
        prob = self.class_head(cls)
        if self.contrastive_head:
            img_emb, txt_emb = self.contrastive_head(img_tokens, cls)
            return prob, img_emb, txt_emb
        return prob

    def encode_text(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(texts, str):
            texts = [texts]
        tokenizer = self.text.tokenizer
        if hasattr(tokenizer, "__call__") and not isinstance(tokenizer, SimpleTokenizer):
            tokens = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            return tokens["input_ids"], tokens["attention_mask"]
        ids = []
        masks = []
        for t in texts:
            out = tokenizer(t)
            ids.append(out["input_ids"].squeeze(0))
            masks.append(out["attention_mask"].squeeze(0))
        return torch.stack(ids), torch.stack(masks)

    def get_last_attention(self) -> torch.Tensor | None:
        return self.fusion.get_last_attention()
