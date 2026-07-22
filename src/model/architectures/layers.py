from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .config import PharmagenConfig, TaskSpec
from .model import PharmagenTwoTower

if TYPE_CHECKING:
    from torch import nn


def create_gnn_model(
    model_name: str,
    drug_config: dict[str, int],
    geno_config: dict[str, int],
    target_dims: dict[str, int],
    params: dict[str, Any],
) -> nn.Module:
    """Bridge the legacy flat-kwargs call onto PharmagenConfig (Phase A shim)."""
    required = ("embedding_dim", "hidden_dim", "dropout_rate", "n_layers", "heads")
    missing = [k for k in required if k not in params]
    if missing:
        raise KeyError(f"Missing model parameters: {missing}")

    cfg = PharmagenConfig(
        drug_in_features=drug_config["num_features"],
        drug_edge_dim=drug_config.get("edge_dim") or None,
        drug_hidden_dim=params["hidden_dim"],
        drug_global_dim=drug_config.get("global_dim", 0),
        drug_admet_dim=drug_config.get("admet_dim", 0),
        geno_in_features=geno_config["num_features"],
        geno_edge_dim=geno_config.get("edge_dim") or None,
        geno_hidden_dim=params["hidden_dim"],
        geno_global_dim=geno_config.get("global_dim", 0),
        embedding_dim=params["embedding_dim"],
        num_layers=params["n_layers"],
        heads=params["heads"],
        dropout=params["dropout_rate"],
        use_polypharmacy=False,
        use_cross_attention=False,
        targets={
            name: TaskSpec(dim=d, kind="multiclass") for name, d in target_dims.items()
        },
    )
    return PharmagenTwoTower(cfg)
