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


def test_multi_binary_axis_excluded_from_embeddings():
    """A binary axis with dim>1 (multi-label) is head-only: no class embedding,
    not composable."""
    axes = {
        "pheno": AxisSpec(name="pheno", dim=3, kind="multiclass", embedding_dim=8),
        "adr": AxisSpec(
            name="adr", dim=5, kind="binary", embedding_dim=8
        ),  # multi-binary
    }
    heads = AxisHeads(in_dim=16, axes=axes)
    # multi-binary axis still gets a prediction head...
    assert heads(torch.randn(2, 16))["adr"].shape == (2, 5)
    # ...but no class embedding and is not single-label / composable.
    assert "adr" not in heads.axis_embeddings
    assert "adr" not in heads.single_label_axes()
    assert "pheno" in heads.axis_embeddings
