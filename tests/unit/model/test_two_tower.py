"""Tests for PharmagenTwoTower — focused on the per-molecule fusion branches.

The drug tower optionally fuses a per-molecule global descriptor vector
(``global_feats``) and a predicted ADMET profile (``admet_feats``) into its graph
embedding; these tests cover each branch on (dim>0) and off (dim=0).
"""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Batch, Data

from src.model.architectures.layers import create_gnn_model

PARAMS = {
    "embedding_dim": 32,
    "hidden_dim": 16,
    "dropout_rate": 0.1,
    "n_layers": 2,
    "heads": 4,
}
GLOBAL_DIM = 1038
ADMET_DIM = 41
GENO_FUNCTION_DIM = 6
TARGETS = {"phenotype_category": 5}


def _drug_graph(*, with_global: bool, with_admet: bool = False) -> Data:
    g = Data(
        x=torch.randn(4, 61),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        edge_attr=torch.randn(3, 18),
    )
    if with_global:
        g.global_feats = torch.randn(1, GLOBAL_DIM)
    if with_admet:
        g.admet_feats = torch.randn(1, ADMET_DIM)
    return g


def _geno_graph() -> Data:
    return Data(
        x=torch.randn(3, 9),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        edge_attr=torch.randn(2, 3),
    )


def _model(global_dim: int, admet_dim: int = 0):
    return create_gnn_model(
        "TwoTowerGAT",
        drug_config={
            "num_features": 61,
            "edge_dim": 18,
            "global_dim": global_dim,
            "admet_dim": admet_dim,
        },
        geno_config={"num_features": 9, "edge_dim": 3},
        target_dims=TARGETS,
        params=PARAMS,
    )


def _geno_graph_v2(*, with_function: bool) -> Data:
    g = Data(
        x=torch.randn(3, 30),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        edge_attr=torch.randn(2, 2),
    )
    if with_function:
        g.geno_function = torch.randn(1, GENO_FUNCTION_DIM)
    return g


def _geno_model(geno_function_dim: int):
    return create_gnn_model(
        "TwoTowerGAT",
        drug_config={"num_features": 61, "edge_dim": 18},
        geno_config={
            "num_features": 30,
            "edge_dim": 2,
            "global_dim": geno_function_dim,
        },
        target_dims=TARGETS,
        params=PARAMS,
    )


class TestGenoFunctionBranch:
    def test_forward_fuses_geno_function(self) -> None:
        model = _geno_model(GENO_FUNCTION_DIM)
        assert hasattr(model, "geno_global_mlp")
        drug = Batch.from_data_list([_drug_graph(with_global=False) for _ in range(3)])
        geno = Batch.from_data_list(
            [_geno_graph_v2(with_function=True) for _ in range(3)]
        )
        out = model(drug, geno)
        assert out["phenotype_category"].shape == (3, 5)

    def test_geno_function_branch_disabled(self) -> None:
        model = _geno_model(0)
        assert not hasattr(model, "geno_global_mlp")
        drug = Batch.from_data_list([_drug_graph(with_global=False) for _ in range(2)])
        geno = Batch.from_data_list(
            [_geno_graph_v2(with_function=False) for _ in range(2)]
        )
        out = model(drug, geno)
        assert out["phenotype_category"].shape == (2, 5)

    def test_missing_geno_function_raises(self) -> None:
        model = _geno_model(GENO_FUNCTION_DIM)
        drug = Batch.from_data_list([_drug_graph(with_global=False) for _ in range(2)])
        # geno graphs lack ``geno_function`` → the fusion branch must complain.
        geno = Batch.from_data_list(
            [_geno_graph_v2(with_function=False) for _ in range(2)]
        )
        with pytest.raises(ValueError, match="geno_function"):
            model(drug, geno)


class TestGlobalBranch:
    def test_forward_with_global_descriptors(self) -> None:
        model = _model(GLOBAL_DIM)
        assert hasattr(model, "drug_global_mlp")
        drug = Batch.from_data_list([_drug_graph(with_global=True) for _ in range(3)])
        geno = Batch.from_data_list([_geno_graph() for _ in range(3)])
        out = model(drug, geno)
        assert out["phenotype_category"].shape == (3, 5)

    def test_global_branch_disabled(self) -> None:
        model = _model(0)
        assert not hasattr(model, "drug_global_mlp")
        drug = Batch.from_data_list([_drug_graph(with_global=False) for _ in range(2)])
        geno = Batch.from_data_list([_geno_graph() for _ in range(2)])
        out = model(drug, geno)
        assert out["phenotype_category"].shape == (2, 5)

    def test_missing_global_feats_raises(self) -> None:
        model = _model(GLOBAL_DIM)
        drug = Batch.from_data_list([_drug_graph(with_global=False) for _ in range(2)])
        geno = Batch.from_data_list([_geno_graph() for _ in range(2)])
        with pytest.raises(ValueError, match="missing 'global_feats'"):
            model(drug, geno)


class TestAdmetBranch:
    def test_forward_with_admet_profile(self) -> None:
        model = _model(0, ADMET_DIM)
        assert hasattr(model, "drug_admet_mlp")
        drug = Batch.from_data_list(
            [_drug_graph(with_global=False, with_admet=True) for _ in range(3)]
        )
        geno = Batch.from_data_list([_geno_graph() for _ in range(3)])
        out = model(drug, geno)
        assert out["phenotype_category"].shape == (3, 5)

    def test_admet_branch_disabled(self) -> None:
        model = _model(0, 0)
        assert not hasattr(model, "drug_admet_mlp")
        drug = Batch.from_data_list([_drug_graph(with_global=False) for _ in range(2)])
        geno = Batch.from_data_list([_geno_graph() for _ in range(2)])
        out = model(drug, geno)
        assert out["phenotype_category"].shape == (2, 5)

    def test_missing_admet_feats_raises(self) -> None:
        model = _model(0, ADMET_DIM)
        drug = Batch.from_data_list(
            [_drug_graph(with_global=False, with_admet=False) for _ in range(2)]
        )
        geno = Batch.from_data_list([_geno_graph() for _ in range(2)])
        with pytest.raises(ValueError, match="missing 'admet_feats'"):
            model(drug, geno)

    def test_global_and_admet_fuse_together(self) -> None:
        model = _model(GLOBAL_DIM, ADMET_DIM)
        assert hasattr(model, "drug_global_mlp")
        assert hasattr(model, "drug_admet_mlp")
        drug = Batch.from_data_list(
            [_drug_graph(with_global=True, with_admet=True) for _ in range(2)]
        )
        geno = Batch.from_data_list([_geno_graph() for _ in range(2)])
        out = model(drug, geno)
        assert out["phenotype_category"].shape == (2, 5)
