"""Flamingo-inspired vision-text architecture."""

from collections import deque
from typing import List, Dict, Optional

import numpy as np

from detector.yolo_detector import YOLODetector

try:
    from captioner.generate_caption import CaptionGenerator
except Exception:  # pragma: no cover - optional dependency
    CaptionGenerator = None  # type: ignore


class FlamingoVisionTextModel:
    """Maintain temporal visual context and generate captions."""

    def __init__(
        self,
        detector: Optional[YOLODetector] = None,
        captioner: Optional[CaptionGenerator] = None,
        context_size: int = 4,
    ) -> None:
        self.detector = detector or YOLODetector()
        self.captioner = captioner
        self.context_size = context_size
        self._frames: deque[np.ndarray] = deque(maxlen=context_size)

    def _aggregate_context(self) -> np.ndarray:
        """Average recent frames with stronger weight to newer ones."""
        frames = list(self._frames)
        if not frames:
            raise ValueError("No frames available for aggregation")

        weights = np.linspace(1.0, 2.0, len(frames), dtype=np.float32)
        stacked = np.stack([f.astype(np.float32) * w for f, w in zip(frames, weights)], axis=0)
        aggregated = stacked.sum(axis=0) / weights.sum()
        return aggregated.astype(np.uint8)

    def process(self, frame: np.ndarray) -> tuple[List[Dict], str]:
        """Detect objects and produce a context-aware caption."""
        detections = self.detector.detect(frame)
        self._frames.append(frame)

        caption = ""
        if self.captioner:
            context_img = self._aggregate_context()
            caption = self.captioner.caption(context_img)
        return detections, caption
