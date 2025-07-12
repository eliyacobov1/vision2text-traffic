import numpy as np
import pytest

pytest.importorskip("torch")

from captioner.transformer_captioner import VisionLanguageModel


def test_vl_generate():
    model = VisionLanguageModel()
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    text = model.generate(img)
    assert isinstance(text, str)
    assert len(text) > 0
