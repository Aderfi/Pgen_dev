# Pharmagen - Modeling
# Architecture: Two-Tower Graph Neural Network (GNN)
# Tower A: Drug (Molecular Graph)
# Tower B: Haplotype/Genome (Interaction Graph using GATv2)

import logging
from typing import Any

from torch import nn

from .gnn import PharmagenTwoTower

logger = logging.getLogger(__name__)

MODEL_ARCHITECTURES = {
    "DeepFM": None,  # Placeholder for other models
    "TwoTowerGAT": PharmagenTwoTower,
}


def create_model(
    model_name: str,  # 'TwoTowerGAT'
    drug_config: dict[str, int],  # {num_features, edge_dim}
    haplo_config: dict[str, int],  # {num_features, edge_dim}
    target_dims: dict[str, int],
    params: dict[str, Any],
) -> nn.Module:
    """
    Factory simplificada para instanciar el modelo Two-Tower.
    """

    # Extracción segura de parámetros con valores por defecto
    try:
        embedding_dim: int = params.get("embedding_dim")
        hidden_dim: int = params.get("hidden_dim")
        dropout: float = params.get("dropout_rate")
        layers: int = params.get("n_layers")
        heads: int = params.get("heads")
    except KeyError as e:
        raise KeyError(f"Missing model parameter: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error: {e}")

    return PharmagenTwoTower(
        drug_in_features=drug_config["num_features"],
        drug_edge_dim=drug_config.get("edge_dim", 0),
        drug_hidden_dim=hidden_dim,
        haplo_in_features=haplo_config["num_features"],
        haplo_edge_dim=haplo_config.get("edge_dim", 0),
        haplo_hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        target_dims=target_dims,
        num_layers=layers,
        heads=heads,
        dropout=dropout,
    )


def create_gnn_model(
    model_name: str,  # 'TwoTowerGAT'
    drug_config: dict[str, int],  # {num_features, edge_dim}
    haplo_config: dict[str, int],  # {num_features, edge_dim}
    target_dims: dict[str, int],
    params: dict[str, Any],
) -> nn.Module:
    """
    Factory simplificada para instanciar el modelo Two-Tower.
    """

    try:
        embedding_dim: int = params.get("embedding_dim")
        hidden_dim: int = params.get("hidden_dim")
        dropout: float = params.get("dropout_rate")
        layers: int = params.get("n_layers")
        heads: int = params.get("heads")

    except KeyError as e:
        raise KeyError(f"Missing model parameter: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error: {e}")

    return PharmagenTwoTower(
        drug_in_features=drug_config["num_features"],
        drug_edge_dim=drug_config.get("edge_dim", 0),
        drug_hidden_dim=hidden_dim,
        haplo_in_features=haplo_config["num_features"],
        haplo_edge_dim=haplo_config.get("edge_dim", 0),
        haplo_hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        target_dims=target_dims,
        num_layers=layers,
        heads=heads,
        dropout=dropout,
    )
