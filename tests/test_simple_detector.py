import numpy as np
import cv2

from detector.simple_detector import SimpleMotionDetector


def test_simple_motion_detector(tmp_path):
    det = SimpleMotionDetector(threshold=10, min_area=1)
    # create two simple frames with a moving square
    f1 = np.zeros((20, 20, 3), dtype=np.uint8)
    f2 = np.zeros((20, 20, 3), dtype=np.uint8)
    cv2.rectangle(f2, (5, 5), (10, 10), (255, 255, 255), -1)
    out1 = det.detect(f1)
    out2 = det.detect(f2)
    assert out1 == []
    assert len(out2) == 1
    assert out2[0]["label"] == "motion"
