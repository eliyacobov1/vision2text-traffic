"""Rule-based caption generator implemented from scratch."""

from typing import List, Dict


class SimpleCaptioner:
    """Generate very basic captions using heuristics."""

    def caption(self, frame, detections: List[Dict]) -> str:
        num_cars = sum(1 for d in detections if d.get("label") == "car")
        if num_cars == 0:
            return "No cars visible."
        if num_cars > 5:
            return "Heavy traffic with many cars."
        if num_cars > 1:
            return "Several cars on the road."
        return "A single car in view."
