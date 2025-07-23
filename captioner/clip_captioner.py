"""CLIP-inspired captioner for traffic scenes."""

from __future__ import annotations

from typing import List

import cv2
import numpy as np

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore


if torch is not None:

    class _TinyCLIP(nn.Module):
        """Minimal CLIP-style model with image and text encoders."""

        def __init__(self, embed_dim: int = 64) -> None:
            super().__init__()
            self.image_encoder = nn.Sequential(
                nn.Conv2d(3, 8, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(8, embed_dim),
            )
            self.text_encoder = nn.Embedding(1000, embed_dim)

        def encode_image(self, x: torch.Tensor) -> torch.Tensor:
            feat = self.image_encoder(x.float() / 255.0)
            return F.normalize(feat, dim=-1)

        def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
            feat = self.text_encoder(tokens)
            return F.normalize(feat, dim=-1)


class CLIPCaptioner:
    """Generate captions by matching prompts using a TinyCLIP model."""

    def __init__(self, model_path: str = "", device: str = "cpu") -> None:
        if torch is None:
            raise ImportError("PyTorch is required for CLIPCaptioner")
        self.model = _TinyCLIP().to(device)
        self.device = device
        self.prompts: List[str] = [
            "heavy traffic",
            "light traffic",
            "pedestrian crossing",
            "accident scene",
            "clear road",
        ]
        self.tokens = torch.arange(len(self.prompts), device=device)

    def caption(self, frame: np.ndarray) -> str:
        tensor = (
            torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
        )
        img_feat = self.model.encode_image(tensor)
        text_feat = self.model.encode_text(self.tokens)
        sim = img_feat @ text_feat.T
        best = int(sim.argmax())
        return self.prompts[best]
