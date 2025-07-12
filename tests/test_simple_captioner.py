import numpy as np

from captioner.simple_captioner import SimpleCaptioner


def test_simple_captioner():
    cap = SimpleCaptioner()
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    detections = [{"label": "car"}] * 3
    text = cap.caption(frame, detections)
    assert "car" in text
