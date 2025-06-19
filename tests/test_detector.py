import numpy as np
import torch
import types

from detector.yolo_detector import YOLODetector


def test_yolo_detector(monkeypatch):
    class DummyTensor(list):
        def tolist(self):
            return self

    class DummyModel:
        def __init__(self):
            self.names = {0: "car"}
        def to(self, device):
            return self
        def __call__(self, frame):
            return types.SimpleNamespace(xyxy=[DummyTensor([[10, 10, 20, 20, 0.9, 0]])])

    monkeypatch.setattr(torch.hub, "load", lambda *a, **k: DummyModel())

    det = YOLODetector("fake.pt", device="cpu")
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    out = det.detect(frame)
    assert out == [{"bbox": [10, 10, 20, 20], "conf": 0.9, "label": "car"}]
