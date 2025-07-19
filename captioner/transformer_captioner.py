"""Simple vision-language transformer implemented from scratch."""

from __future__ import annotations

from typing import List

import cv2

try:  # optional dependency
    import torch
    from torch import nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - torch may be missing
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
import numpy as np


if torch is not None:

    class CNNEncoder(nn.Module):
        def __init__(self, out_dim: int = 128) -> None:
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )
            self.fc = nn.Linear(64, out_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            feat = self.conv(x)
            feat = feat.view(x.size(0), -1)
            return self.fc(feat)


def _positional_encoding(length: int, dim: int) -> torch.Tensor:
    pe = torch.zeros(length, dim)
    pos = torch.arange(0, length, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32) * -(np.log(10000.0) / dim)
    )
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


if torch is not None:

    class VisionLanguageModel(nn.Module):
        """Transformer-based captioner.

        The model pairs a small CNN encoder with a decoder-only transformer.
        This mirrors the structure used in many vision-language systems where
        image features are embedded and then decoded into text. The
        implementation is intentionally minimal yet highlights the key
        components required for scene description tasks.
        """

        def __init__(self, vocab_size: int = 100, hidden_dim: int = 128, num_layers: int = 2) -> None:
            super().__init__()
            self.encoder = CNNEncoder(hidden_dim)
            self.token_emb = nn.Embedding(vocab_size, hidden_dim)
            decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=4)
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
            self.fc_out = nn.Linear(hidden_dim, vocab_size)
            self.register_buffer("pos", _positional_encoding(50, hidden_dim))
            self.sos = 1
            self.eos = 2
            self.vocab = {i: f"tok{i}" for i in range(vocab_size)}
            self.vocab[self.sos] = "<s>"
            self.vocab[self.eos] = "</s>"

        def forward(self, images: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
            # captions: (B, T)
            feat = self.encoder(images).unsqueeze(0)
            T = captions.size(1)
            emb = self.token_emb(captions).permute(1, 0, 2)
            emb = emb + self.pos[:T].unsqueeze(1)
            output = self.decoder(emb[:-1], feat)
            logits = self.fc_out(output)
            return logits

        @torch.no_grad()
        def generate(
            self,
            image: np.ndarray,
            boxes: List[List[int]] | None = None,
            max_len: int = 20,
            device: str = "cpu",
        ) -> str:
            """Generate caption optionally using object region features.

            Args:
                image: BGR image array.
                boxes: Optional list of ``[x1, y1, x2, y2]`` regions describing
                    detected objects. When provided, features from these regions
                    are averaged with the global image feature before decoding.
                max_len: Maximum length of generated caption.
                device: Device string for computation.

            Returns:
                Generated caption string.
            """

            self.eval()
            tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0) / 255
            tensor = tensor.to(device)
            memory = [self.encoder(tensor)]

            if boxes:
                crops = []
                h, w, _ = image.shape
                for b in boxes:
                    x1, y1, x2, y2 = map(int, b)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    crop = image[y1:y2, x1:x2]
                    if crop.size != 0:
                        crop = cv2.resize(crop, (image.shape[1], image.shape[0]))
                        crops.append(crop)
                if crops:
                    roi_tensor = (
                        torch.from_numpy(np.stack(crops))
                        .permute(0, 3, 1, 2)
                        .float()
                        / 255
                    )
                    roi_tensor = roi_tensor.to(device)
                    region_feat = self.encoder(roi_tensor).mean(0, keepdim=True)
                    memory.append(region_feat)

            memory_t = torch.stack(memory).to(device)  # (M,B,H)

            ys = torch.tensor([[self.sos]], device=device)
            for _ in range(max_len):
                emb = self.token_emb(ys).permute(1, 0, 2) + self.pos[: ys.size(1)].unsqueeze(1)
                out = self.decoder(emb, memory_t)
                prob = self.fc_out(out[-1])
                next_tok = prob.argmax(dim=-1)
                ys = torch.cat([ys, next_tok.unsqueeze(0)], dim=1)
                if next_tok.item() == self.eos:
                    break

            tokens = [self.vocab.get(t.item(), "?") for t in ys[0, 1:]]
            return " ".join(tokens)
else:

    def _positional_encoding(length: int, dim: int) -> torch.Tensor:
        raise ImportError("PyTorch is required for positional encoding")

    class VisionLanguageModel:
        """Placeholder when PyTorch is unavailable."""

        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("PyTorch is required for VisionLanguageModel")
