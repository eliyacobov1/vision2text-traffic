"""Traffic congestion detection logic."""

from typing import List, Dict, Optional

import numpy as np


class CongestionDetector:
    """Simple congestion detector based on vehicle density and movement."""

    def __init__(self, displacement_threshold: float = 2.0, vehicle_count_threshold: int = 5) -> None:
        self.displacement_threshold = displacement_threshold
        self.vehicle_count_threshold = vehicle_count_threshold
        self._prev_centroids: Optional[List[np.ndarray]] = None

    @staticmethod
    def _centroid(box: List[float]) -> np.ndarray:
        x1, y1, x2, y2 = box
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

    def update(self, detections: List[Dict]) -> bool:
        """Update detector with the latest frame's detections.

        Args:
            detections: List of detection dictionaries from ``YOLODetector``.

        Returns:
            ``True`` if congestion is detected, else ``False``.
        """
        centroids = [self._centroid(det["bbox"]) for det in detections]
        if self._prev_centroids is None:
            self._prev_centroids = centroids
            return False

        displacements = []
        for c in centroids:
            if not self._prev_centroids:
                continue
            distances = [np.linalg.norm(c - p) for p in self._prev_centroids]
            if distances:
                displacements.append(min(distances))
        avg_disp = np.mean(displacements) if displacements else float("inf")

        congested = avg_disp < self.displacement_threshold and len(centroids) >= self.vehicle_count_threshold

        self._prev_centroids = centroids
        return congested
