"""Smoke test exercising the split-out PharmagenTwoTower forward pass.

This complements the golden structural test: it drives a full forward pass
through the model built purely from ``src.model.architectures`` (config.py +
model.py + the already-split towers/fusion subpackages) to catch wiring
regressions introduced by the A5 module split.
"""

from __future__ import annotations

import torch
from torch_geometric.data import Batch, Data

from src.model.architectures import PharmagenConfig, PharmagenTwoTower, TaskSpec


def _drug_graph() -> Data:
    # GINEConv requires edge_attr, so the tower is always exercised with edges.
    return Data(
        x=torch.randn(4, 6),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        edge_attr=torch.randn(3, 4),
    )


def _geno_graph() -> Data:
    return Data(
        x=torch.randn(3, 5),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        edge_attr=torch.randn(2, 3),
    )


def _cfg() -> PharmagenConfig:
    return PharmagenConfig(
        drug_in_features=6,
        drug_edge_dim=4,
        drug_hidden_dim=16,
        geno_in_features=5,
        geno_edge_dim=3,
        geno_hidden_dim=16,
        embedding_dim=16,
        num_layers=2,
        heads=2,
        dropout=0.0,
        use_polypharmacy=False,
        use_cross_attention=False,
        axes={"outcome": TaskSpec(dim=1, kind="binary")},
    )


def test_forward_pass_produces_expected_output_shape():
    model = PharmagenTwoTower(_cfg())
    drug_batch = Batch.from_data_list([_drug_graph()])
    geno_batch = Batch.from_data_list([_geno_graph()])

    outputs = model(drug_batch, geno_batch)

    # `outcome` is a binary axis (not composable), so with no composable axis
    # the compositional "_z" output is disabled — only the per-axis logit.
    assert set(outputs) == {"outcome"}
    assert outputs["outcome"].shape == (1, 1)


def test_imports_do_not_touch_deleted_gnn_module():
    from src.model import architectures

    assert not hasattr(architectures, "gnn")
