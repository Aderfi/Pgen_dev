"""Bridge test: create_gnn_model must build a PharmagenConfig-based model."""

from __future__ import annotations

from src.model.architectures import PharmagenTwoTower
from src.model.architectures.layers import create_gnn_model


def test_bridge_builds_two_tower():
    model = create_gnn_model(
        model_name="TwoTowerGAT",
        drug_config={"num_features": 6, "edge_dim": 0, "global_dim": 0, "admet_dim": 0},
        geno_config={"num_features": 5, "edge_dim": 0, "global_dim": 0},
        target_dims={"pheno": 3},
        params={
            "embedding_dim": 16,
            "hidden_dim": 16,
            "dropout_rate": 0.0,
            "n_layers": 2,
            "heads": 2,
        },
    )
    assert isinstance(model, PharmagenTwoTower)
