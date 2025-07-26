import sys, os; sys.path.append(os.getcwd())
import torch
from model import VisionLanguageTransformer, VLTConfig


def test_forward_classification():
    cfg = VLTConfig()
    model = VisionLanguageTransformer(cfg)
    images = torch.randn(2, 3, 224, 224)
    input_ids = torch.zeros(2, 8, dtype=torch.long)
    mask = torch.ones_like(input_ids)
    out = model(images, input_ids, mask)
    assert out.shape == (2,)


def test_config_load():
    cfg = VLTConfig(hidden_dim=128)
    assert cfg.hidden_dim == 128
