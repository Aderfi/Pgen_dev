import torch

from src.model.architectures.config import AxisSpec
from src.model.architectures.heads.axis_heads import AxisHeads
from src.model.architectures.heads.compose import ComposeHead


def test_compose_produces_label_vector_and_is_differentiable():
    axes = {"pheno": AxisSpec(name="pheno", dim=3, kind="multiclass", embedding_dim=8)}
    heads = AxisHeads(in_dim=16, axes=axes)
    compose = ComposeHead(axes=axes, out_dim=12)
    z = torch.randn(4, 16, requires_grad=True)
    logits = heads(z)
    out = compose(logits, heads.axis_embeddings)
    assert out.shape == (4, 12)
    out.sum().backward()
    assert z.grad is not None


def test_embed_tuples_matches_dim():
    axes = {"pheno": AxisSpec(name="pheno", dim=3, kind="multiclass", embedding_dim=8)}
    heads = AxisHeads(in_dim=16, axes=axes)
    compose = ComposeHead(axes=axes, out_dim=12)
    tuples = torch.tensor([[0], [2]])
    table = compose.embed_tuples(tuples, heads.axis_embeddings)
    assert table.shape == (2, 12)
