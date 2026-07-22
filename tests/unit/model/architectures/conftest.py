"""Shared fixtures for `src.model.architectures` unit tests.

``make_batch`` builds the smallest possible pair of PyG ``Batch`` objects that
exercise both towers: two patients, one drug (molecule) graph and one
genotype graph each.

GINEConv (the drug tower's conv when ``use_mol_gnn=True``) requires
``edge_attr`` -- it errors on ``None`` -- so every drug graph here carries one.
GATv2 (the genotype tower) accepts ``edge_attr=None``, so the geno graphs
intentionally omit it.
"""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Batch, Data

DRUG_IN_FEATURES = 6
DRUG_EDGE_DIM = 4
GENO_IN_FEATURES = 5


def _drug_graph() -> Data:
    return Data(
        x=torch.randn(4, DRUG_IN_FEATURES),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        edge_attr=torch.randn(3, DRUG_EDGE_DIM),
    )


def _geno_graph() -> Data:
    return Data(
        x=torch.randn(3, GENO_IN_FEATURES),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
    )


@pytest.fixture
def make_batch():
    """Return a factory building `(drug_batch, geno_batch)` for 2 patients."""

    def _make() -> tuple[Batch, Batch]:
        drug_batch = Batch.from_data_list([_drug_graph(), _drug_graph()])
        geno_batch = Batch.from_data_list([_geno_graph(), _geno_graph()])
        return drug_batch, geno_batch

    return _make
