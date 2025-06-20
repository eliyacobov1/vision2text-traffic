import numpy as np
from flamingo.vision_text_model import FlamingoVisionTextModel


def test_flamingo_process():
    class DummyDetector:
        def detect(self, frame):
            return [{"bbox": [0, 0, 1, 1], "conf": 0.9, "label": "car"}]

    class DummyCaption:
        def caption(self, frame):
            return "dummy"

    captured = {}

    class DummyCaption:
        def caption(self, frame):
            captured["frame"] = frame.copy()
            return "dummy"

    model = FlamingoVisionTextModel(DummyDetector(), DummyCaption(), context_size=2)
    f1 = np.zeros((2, 2, 3), dtype=np.uint8)
    f2 = np.ones((2, 2, 3), dtype=np.uint8) * 255
    model.process(f1)
    model.process(f2)
    det, cap = model.process(f2)

    assert det[0]["label"] == "car"
    assert cap == "dummy"
    # The aggregated frame should be weighted toward the newest frame (f2).
    assert captured["frame"].mean() > 100
