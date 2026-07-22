"""Compositional-label decode/agreement wiring in ``PGenPredictor``.

Exercises ``PGenPredictor._attach_composed_labels`` directly against real
``AxisHeads`` / ``ComposeHead`` / ``CompositionalLabelTable`` objects — the
same pieces ``__init__`` wires up from the persisted bundle — without
needing a full trained checkpoint or graph library on disk.
"""

from __future__ import annotations

import torch

from src.model.architectures.config import AxisSpec
from src.model.architectures.heads.axis_heads import AxisHeads
from src.model.architectures.heads.compose import ComposeHead
from src.model.architectures.heads.label_table import CompositionalLabelTable
from src.model.engine.predictor import PGenPredictor


def _build_compose_fixture():
    axes = {
        "dir": AxisSpec(name="dir", dim=2, kind="ordinal", embedding_dim=6),
        "pheno": AxisSpec(name="pheno", dim=3, kind="multiclass", embedding_dim=6),
    }
    heads = AxisHeads(in_dim=16, axes=axes)
    compose = ComposeHead(axes=axes, out_dim=10)
    order = compose.single_label_axes()
    assert order == ["dir", "pheno"]

    tuples = [(0, 0), (1, 2), (0, 1)]
    table = CompositionalLabelTable(tuples=tuples, labels=["x", "y", "z"])
    table.build(compose, heads.axis_embeddings)
    return heads, compose, table


def test_attach_composed_labels_adds_top1_topk_and_agreement():
    heads, compose, table = _build_compose_fixture()

    # Confident logits that argmax to (1, 2) -> exactly table row "y".
    logits = {
        "dir": torch.tensor([[-5.0, 5.0]]),
        "pheno": torch.tensor([[-5.0, -5.0, 5.0]]),
    }
    z = compose(logits, heads.axis_embeddings)

    predictor = PGenPredictor.__new__(PGenPredictor)
    predictor.label_table = table
    predictor._composable_axes = compose.single_label_axes()

    results = [{}]
    predictor._attach_composed_labels(results, logits, z)

    assert results[0]["composed_label"] == "y"
    assert results[0]["composed_agreement"] is True
    assert results[0]["composed_topk"][0][0] == "y"
    assert all(isinstance(score, float) for _, score in results[0]["composed_topk"])


def test_attach_composed_labels_disagreement_is_false():
    heads, compose, table = _build_compose_fixture()

    # z from row "x" (tuple (0, 0)) but argmax logits point at (1, 2).
    logits = {
        "dir": torch.tensor([[-5.0, 5.0]]),
        "pheno": torch.tensor([[-5.0, -5.0, 5.0]]),
    }
    x_tuple = torch.as_tensor([[0, 0]], dtype=torch.long)
    z = compose.embed_tuples(x_tuple, heads.axis_embeddings)

    predictor = PGenPredictor.__new__(PGenPredictor)
    predictor.label_table = table
    predictor._composable_axes = compose.single_label_axes()

    results = [{}]
    predictor._attach_composed_labels(results, logits, z)

    assert results[0]["composed_label"] == "x"
    assert results[0]["composed_agreement"] is False


def test_decode_logits_without_label_table_is_unaffected():
    """No composed keys are added when the predictor has no label table."""
    predictor = PGenPredictor.__new__(PGenPredictor)
    predictor.label_table = None
    predictor.target_cols = ["pheno"]
    predictor.multi_label_cols = set()

    class _FakeEncoder:
        def inverse_transform(self, idx):
            return ["poor" for _ in idx]

    predictor.encoders = {"pheno": _FakeEncoder()}

    logits = {"pheno": torch.tensor([[1.0, 0.0, 0.0]])}
    results = predictor._decode_logits(logits)

    assert results == [{"pheno": "poor"}]
    assert "composed_label" not in results[0]
