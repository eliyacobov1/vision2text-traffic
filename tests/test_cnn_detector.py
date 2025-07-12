import numpy as np
import pytest

pytest.importorskip("torch")

from detector.cnn_detector import CNNDetector


def test_cnn_detector():
    det = CNNDetector()
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    out = det.detect(frame)
    assert isinstance(out, list)
