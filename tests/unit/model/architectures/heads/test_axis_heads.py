import torch

from src.model.architectures.config import AxisSpec
from src.model.architectures.heads.axis_heads import AxisHeads


def test_axis_heads_emit_per_axis_logits():
    axes = {
        "pheno": AxisSpec(name="pheno", dim=3, kind="multiclass", embedding_dim=8),
        "assoc": AxisSpec(name="assoc", dim=1, kind="binary", embedding_dim=8),
    }
    heads = AxisHeads(in_dim=16, axes=axes)
    out = heads(torch.randn(4, 16))
    assert out["pheno"].shape == (4, 3) and out["assoc"].shape == (4, 1)
