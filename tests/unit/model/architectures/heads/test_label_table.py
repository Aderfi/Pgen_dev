import torch

from src.model.architectures.config import AxisSpec
from src.model.architectures.heads.axis_heads import AxisHeads
from src.model.architectures.heads.compose import ComposeHead
from src.model.architectures.heads.label_table import CompositionalLabelTable


def test_decode_returns_topk_labels():
    axes = {"pheno": AxisSpec(name="pheno", dim=3, kind="multiclass", embedding_dim=8)}
    heads = AxisHeads(in_dim=16, axes=axes)
    compose = ComposeHead(axes=axes, out_dim=12)
    table = CompositionalLabelTable(tuples=[(0,), (1,), (2,)], labels=["A", "B", "C"])
    emb = table.build(compose, heads.axis_embeddings)
    assert emb.shape == (3, 12)
    z = emb[1:2]  # exactly the "B" row
    decoded = table.decode(z, top_k=2)
    assert decoded[0][0][0] == "B"


def test_multi_axis_agreement_and_decode():
    """Two single-label axes: exercise column ordering + agreement()."""
    axes = {
        "dir": AxisSpec(name="dir", dim=2, kind="ordinal", embedding_dim=6),
        "pheno": AxisSpec(name="pheno", dim=3, kind="multiclass", embedding_dim=6),
    }
    heads = AxisHeads(in_dim=16, axes=axes)
    compose = ComposeHead(axes=axes, out_dim=10)
    order = compose.single_label_axes()  # column order for tuples
    assert set(order) == {"dir", "pheno"}
    tuples = [(0, 0), (1, 2), (0, 1)]
    table = CompositionalLabelTable(tuples=tuples, labels=["x", "y", "z"])
    emb = table.build(compose, heads.axis_embeddings)
    assert emb.shape == (3, 10)

    # Decoding an exact row returns its label as top-1.
    assert table.decode(emb[1:2], top_k=1)[0][0][0] == "y"

    # agreement: nearest row's tuple == supplied argmax tuple -> True; a
    # non-table tuple for the same z -> False.
    argmax_match = torch.tensor([tuples[1]])  # (1, 2)
    argmax_wrong = torch.tensor([[0, 0]])
    assert table.agreement(emb[1:2], argmax_match).tolist() == [True]
    assert table.agreement(emb[1:2], argmax_wrong).tolist() == [False]
