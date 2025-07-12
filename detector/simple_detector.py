"""Basic motion-based detector implemented from scratch."""

from typing import List, Dict, Optional

import cv2
import numpy as np


class SimpleMotionDetector:
    """Detect moving regions by frame differencing."""

    def __init__(self, threshold: int = 25, min_area: int = 500) -> None:
        self.prev_gray: Optional[np.ndarray] = None
        self.threshold = threshold
        self.min_area = min_area

    def detect(self, frame: np.ndarray) -> List[Dict]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections: List[Dict] = []
        if self.prev_gray is not None:
            diff = cv2.absdiff(self.prev_gray, gray)
            _, mask = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for cnt in contours:
                if cv2.contourArea(cnt) >= self.min_area:
                    x, y, w, h = cv2.boundingRect(cnt)
                    detections.append({
                        "bbox": [x, y, x + w, y + h],
                        "conf": 1.0,
                        "label": "motion",
                    })
        self.prev_gray = gray
        return detections
