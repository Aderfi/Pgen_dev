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
