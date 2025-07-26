import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple, List

from encoders import VisionEncoder, TextEncoder, SimpleTokenizer
from fusion import CrossModalFusion
from heads import get_head, ClassificationHead, ContrastiveHead


@dataclass
class VLTConfig:
    vision_model: str = "vit_base_patch16_224"
    text_model: str = "distilbert-base-uncased"
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 2
    contrastive: bool = False
    loss_type: str = "classification"  # classification | contrastive | hybrid
    heads: Tuple[str, ...] = ("classification",)  # list of heads to include


class VisionLanguageTransformer(nn.Module):
    def __init__(self, config: VLTConfig = VLTConfig(), offline: bool = True):
        super().__init__()
        self.config = config
        self.vision = VisionEncoder(config.vision_model, freeze=True)
        self.text = TextEncoder(config.text_model, freeze=True, offline=offline)

        self.img_proj = nn.Linear(self.vision.out_dim, config.hidden_dim)
        self.txt_proj = nn.Linear(self.text.out_dim, config.hidden_dim)

        self.fusion = CrossModalFusion(config.hidden_dim, config.num_heads, config.num_layers)

        self.heads = nn.ModuleDict()
        for name in config.heads:
            self.heads[name] = get_head(name, config.hidden_dim)
        if config.contrastive and "contrastive" not in self.heads:
            self.heads["contrastive"] = ContrastiveHead(config.hidden_dim)
        if "classification" not in self.heads:
            self.heads["classification"] = ClassificationHead(config.hidden_dim)

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        img_tokens = self.img_proj(self.vision(images))
        txt_tokens = self.txt_proj(self.text(input_ids, attention_mask))
        fused = self.fusion(txt_tokens, img_tokens)
        cls = fused[:, 0, :]

        outputs = {}
        for name, head in self.heads.items():
            if name == "contrastive":
                outputs[name] = head(img_tokens, cls)
            else:
                outputs[name] = head(cls)
        if len(outputs) == 1:
            return next(iter(outputs.values()))
        return outputs

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
