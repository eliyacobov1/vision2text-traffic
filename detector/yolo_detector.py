"""YOLO-based vehicle detector."""

from typing import List, Dict

import numpy as np

try:  # optional dependency
    import torch
except Exception:  # pragma: no cover - torch may be missing
    torch = None  # type: ignore


class YOLODetector:
    """Wrapper around a YOLO model for vehicle detection."""

    def __init__(self, model_path: str = "yolov5s.pt", device: str = "cpu") -> None:
        """Load YOLOv5 model via ``torch.hub``.

        Args:
            model_path: Path to the YOLOv5 weights.
            device: Device string for computation (e.g., ``"cpu"`` or ``"cuda"``).
        """
        if torch is None:
            raise ImportError("PyTorch is required for YOLODetector")
        self.model = torch.hub.load(
            "ultralytics/yolov5", "custom", path=model_path, trust_repo=True
        )
        self.model.to(device)
        self.device = device

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """Run detection on a single frame.

        Args:
            frame: BGR image array.

        Returns:
            List of detections with keys ``bbox`` (x1,y1,x2,y2), ``conf`` and ``label``.
        """
        results = self.model(frame)
        detections = []
        names = self.model.names
        for box in results.xyxy[0].tolist():
            x1, y1, x2, y2, conf, cls_id = box
            label = names[int(cls_id)]
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "conf": float(conf),
                "label": label,
            })
        return detections
