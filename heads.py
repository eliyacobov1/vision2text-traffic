# Model head implementations and registry
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Type


class BaseHead(nn.Module):
    """Base class for heads."""

    def forward(self, *inputs):
        raise NotImplementedError


class ClassificationHead(BaseHead):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, 1)

    def forward(self, cls: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc(cls)).squeeze(-1)


class ContrastiveHead(BaseHead):
    def __init__(self, dim: int):
        super().__init__()
        self.img_proj = nn.Linear(dim, dim)
        self.txt_proj = nn.Linear(dim, dim)

    def forward(self, img_tokens: torch.Tensor, txt_cls: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        img_feat = img_tokens.mean(dim=1)
        img_emb = F.normalize(self.img_proj(img_feat), dim=-1)
        txt_emb = F.normalize(self.txt_proj(txt_cls), dim=-1)
        return img_emb, txt_emb


class CaptioningHead(BaseHead):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # placeholder for captioning module

    def forward(self, *args):
        raise NotImplementedError("Captioning head not implemented")


class VQAHead(BaseHead):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # placeholder for VQA module

    def forward(self, *args):
        raise NotImplementedError("VQA head not implemented")


HEAD_REGISTRY: Dict[str, Type[BaseHead]] = {
    "classification": ClassificationHead,
    "contrastive": ContrastiveHead,
    "captioning": CaptioningHead,
    "vqa": VQAHead,
}


def get_head(name: str, dim: int) -> BaseHead:
    if name not in HEAD_REGISTRY:
        raise ValueError(f"Unknown head type: {name}")
    return HEAD_REGISTRY[name](dim)
