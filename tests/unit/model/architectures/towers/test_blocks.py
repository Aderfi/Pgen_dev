import pytest
import torch

from src.model.architectures.towers.blocks import branch_mlp, make_conv, masked_pool


def test_branch_mlp_shapes():
    mlp = branch_mlp(8, 16, dropout=0.0)
    out = mlp(torch.randn(4, 8))
    assert out.shape == (4, 16)


def test_masked_pool_returns_double_width():
    x = torch.ones(2, 3, 5)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    pooled = masked_pool(x, mask)
    assert pooled.shape == (2, 10)  # [sum ; mean] over dim=1


def test_make_conv_rejects_unknown_type():
    with pytest.raises(ValueError, match="conv_type"):
        make_conv("bogus", dim=16, heads=4, dropout=0.0, edge_dim=None)
