import numpy as np
from analyzer.congestion_detector import CongestionDetector


def test_congestion_detection():
    detector = CongestionDetector(displacement_threshold=2.0, vehicle_count_threshold=2)
    detections1 = [
        {"bbox": [10, 10, 20, 20]},
        {"bbox": [30, 30, 40, 40]},
    ]
    assert detector.update(detections1) is False

    detections2 = [
        {"bbox": [11, 10, 21, 20]},
        {"bbox": [31, 30, 41, 40]},
    ]
    assert detector.update(detections2) is True
