"""Convolutional object detector implemented from scratch."""

from __future__ import annotations

from typing import List, Dict

try:  # optional dependency
    import torch
    from torch import nn
except Exception:  # pragma: no cover - torch may be missing
    torch = None  # type: ignore
    nn = None  # type: ignore
import numpy as np
import cv2


if torch is not None:

    class _Conv(nn.Sequential):
        def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1) -> None:
            pad = k // 2
            super().__init__(
                nn.Conv2d(c1, c2, k, s, pad),
                nn.BatchNorm2d(c2),
                nn.ReLU(inplace=True),
            )


    class TinyBackbone(nn.Module):
        """Simple CNN backbone."""

        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.Sequential(
                _Conv(3, 16),
                _Conv(16, 16),
                nn.MaxPool2d(2),
                _Conv(16, 32),
                _Conv(32, 32),
                nn.MaxPool2d(2),
                _Conv(32, 64),
                _Conv(64, 64),
                nn.MaxPool2d(2),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.layers(x)


    class TinyDetectionHead(nn.Module):
        """Predict bounding boxes for each grid cell."""

        def __init__(self, in_channels: int, num_classes: int, grid_size: int = 8) -> None:
            super().__init__()
            self.grid_size = grid_size
            self.num_classes = num_classes
            self.pred = nn.Conv2d(in_channels, (5 + num_classes), 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.pred(x)


    class CNNDetector:
        """Detector with a tiny CNN backbone and head."""

        # This simplified architecture mimics the grid-based design of modern
        # one-stage detectors such as YOLO. It is intentionally lightweight and
        # uses random weights for demonstration purposes, but showcases the
        # typical flow of feature extraction followed by bounding box
        # prediction.

        def __init__(self, num_classes: int = 1, device: str = "cpu") -> None:
            self.device = device
            self.backbone = TinyBackbone().to(device)
            self.head = TinyDetectionHead(64, num_classes).to(device)
            self.classes = ["car"]

        @torch.no_grad()
        def detect(self, frame: np.ndarray) -> List[Dict]:
            """Run detection on a frame using random weights."""
            img = cv2.resize(frame, (64, 64))
            tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255
            tensor = tensor.to(self.device)
            feats = self.backbone(tensor)
            pred = self.head(feats)[0]  # (C,H,W)
            pred = pred.permute(1, 2, 0).cpu().numpy()  # (H,W,C)
            h, w, _ = pred.shape
            boxes: List[Dict] = []
            for iy in range(h):
                for ix in range(w):
                    obj_conf = 1 / (1 + np.exp(-pred[iy, ix, 4]))
                    if obj_conf < 0.5:
                        continue
                    bx, by, bw, bh = pred[iy, ix, :4]
                    bx = (ix + bx) / w
                    by = (iy + by) / h
                    bw = max(bw, 1e-2)
                    bh = max(bh, 1e-2)
                    x1 = max(0, (bx - bw / 2) * frame.shape[1])
                    y1 = max(0, (by - bh / 2) * frame.shape[0])
                    x2 = min(frame.shape[1], (bx + bw / 2) * frame.shape[1])
                    y2 = min(frame.shape[0], (by + bh / 2) * frame.shape[0])
                    label = self.classes[0]
                    boxes.append({
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "conf": float(obj_conf),
                        "label": label,
                    })
            return boxes
else:

    class CNNDetector:
        """Placeholder when PyTorch is unavailable."""

        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("PyTorch is required for CNNDetector")
