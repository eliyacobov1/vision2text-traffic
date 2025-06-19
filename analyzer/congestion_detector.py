"""Traffic congestion detection logic."""

from typing import List, Dict, Optional
from collections import deque

import numpy as np

from tracking import KalmanFilter, lucas_kanade_flow


class CongestionDetector:
    """Simple congestion detector based on vehicle density and movement."""

    def __init__(
        self,
        displacement_threshold: float = 2.0,
        vehicle_count_threshold: int = 5,
        history_frames: int = 3,
        match_distance: float = 30.0,
    ) -> None:
        self.displacement_threshold = displacement_threshold
        self.vehicle_count_threshold = vehicle_count_threshold
        self.history_frames = history_frames
        self.match_distance = match_distance
        self._prev_centroids: Optional[List[np.ndarray]] = None
        self._prev_frame: Optional[np.ndarray] = None
        self._count_history: deque[int] = deque(maxlen=history_frames)
        self._speed_history: deque[float] = deque(maxlen=history_frames)
        self._trackers: List[KalmanFilter] = []
        self._missed: List[int] = []

    @staticmethod
    def _centroid(box: List[float]) -> np.ndarray:
        x1, y1, x2, y2 = box
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

    def update(self, detections: List[Dict], frame: Optional[np.ndarray] = None) -> bool:
        """Update detector with the latest frame's detections.

        Args:
            detections: List of detection dictionaries from ``YOLODetector``.

        Returns:
            ``True`` if congestion is detected, else ``False``.
        """
        centroids = [self._centroid(det["bbox"]) for det in detections]

        if self._prev_centroids is None:
            self._prev_centroids = centroids
            self._prev_frame = frame
            self._count_history.append(len(centroids))
            self._speed_history.append(0.0)
            self._trackers = [KalmanFilter() for _ in centroids]
            for t, c in zip(self._trackers, centroids):
                t.x[:2] = c
            self._missed = [0 for _ in centroids]
            return False

        # Predict step for all trackers
        predictions = [t.predict() for t in self._trackers]

        # Assign detections to trackers greedily
        assigned = [-1] * len(centroids)
        used = set()
        for i, c in enumerate(centroids):
            dists = [np.linalg.norm(c - p) if j not in used else np.inf for j, p in enumerate(predictions)]
            if not dists:
                continue
            j = int(np.argmin(dists))
            if dists[j] < self.match_distance:
                assigned[i] = j
                used.add(j)
                self._trackers[j].update(c)
                self._missed[j] = 0
        # Age and remove unmatched trackers
        for idx in range(len(self._trackers)):
            if idx not in used:
                self._missed[idx] += 1
        # Remove old trackers
        keep = [i for i, m in enumerate(self._missed) if m < self.history_frames]
        self._trackers = [self._trackers[i] for i in keep]
        self._missed = [self._missed[i] for i in keep]

        # Add new trackers for unmatched detections
        for det_idx, trk_idx in enumerate(assigned):
            if trk_idx == -1:
                kf = KalmanFilter()
                kf.x[:2] = centroids[det_idx]
                self._trackers.append(kf)
                self._missed.append(0)

        # Compute optical flow if frames available
        if frame is not None and self._prev_frame is not None and self._prev_centroids:
            prev_gray = np.mean(self._prev_frame, axis=2)
            curr_gray = np.mean(frame, axis=2)
            flows = lucas_kanade_flow(prev_gray, curr_gray, self._prev_centroids)
            speeds = [np.linalg.norm(f) for f in flows]
            avg_speed = float(np.mean(speeds)) if speeds else float("inf")
        else:
            speeds = [np.linalg.norm(c - p) for c, p in zip(centroids, self._prev_centroids)]
            avg_speed = float(np.mean(speeds)) if speeds else float("inf")

        self._prev_centroids = centroids
        self._prev_frame = frame
        self._count_history.append(len(centroids))
        self._speed_history.append(avg_speed)

        if len(self._count_history) < self.history_frames:
            return False

        count_high = all(c >= self.vehicle_count_threshold for c in self._count_history)
        speed_low = all(s < self.displacement_threshold for s in self._speed_history)
        return count_high and speed_low
