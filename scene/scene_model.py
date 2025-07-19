"""Unified scene understanding model combining detection and captioning."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from detector.yolo_detector import YOLODetector

try:
    from captioner.generate_caption import CaptionGenerator
    from captioner.simple_captioner import SimpleCaptioner
    from captioner.transformer_captioner import VisionLanguageModel
except Exception:  # pragma: no cover - captioning is optional
    CaptionGenerator = None  # type: ignore
    SimpleCaptioner = None  # type: ignore
    VisionLanguageModel = None  # type: ignore


class SceneUnderstandingModel:
    """High level scene analysis similar to Gemini's approach."""

    def __init__(
        self,
        detector: Optional[YOLODetector] = None,
        captioner: Optional[object] = None,
    ) -> None:
        self.detector = detector or YOLODetector()
        if captioner is None and CaptionGenerator:
            captioner = CaptionGenerator()
        self.captioner = captioner

    def understand(self, frame: np.ndarray) -> Dict[str, Any]:
        """Return detections and a scene level caption."""
        detections = self.detector.detect(frame)
        caption = ""
        if self.captioner:
            if hasattr(self.captioner, "generate"):
                boxes = [d["bbox"] for d in detections]
                caption = self.captioner.generate(frame, boxes=boxes)
            else:
                caption = self.captioner.caption(frame)
        return {"detections": detections, "caption": caption}

