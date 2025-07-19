import numpy as np
from scene.scene_model import SceneUnderstandingModel


def test_scene_understanding():
    class DummyDetector:
        def detect(self, frame):
            return [{"bbox": [0, 0, 1, 1], "conf": 0.9, "label": "car"}]

    class DummyCaption:
        def caption(self, frame):
            return "dummy"

    model = SceneUnderstandingModel(DummyDetector(), DummyCaption())
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    result = model.understand(img)
    assert result["detections"][0]["label"] == "car"
    assert result["caption"] == "dummy"

