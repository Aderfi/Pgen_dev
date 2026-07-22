"""``assembly.create_gnn_model`` owns model construction from dims + axes."""

from __future__ import annotations

from src.model.architectures import PharmagenTwoTower
from src.model.architectures.assembly import create_gnn_model
from src.model.architectures.config import AxisSpec


def test_create_from_axes():
    model = create_gnn_model(
        dims={
            "drugs": {"edges": 0, "global": 0, "admet": 0},
            "geno": {"edges": 0, "function": 0},
        },
        drug_dim=6,
        geno_dim=5,
        axes={"pheno": AxisSpec(name="pheno", dim=3, kind="multiclass")},
        params={
            "embedding_dim": 16,
            "hidden_dim": 16,
            "dropout_rate": 0.0,
            "n_layers": 2,
            "heads": 2,
        },
        switches={"use_polypharmacy": False, "use_cross_attention": False},
    )
    assert isinstance(model, PharmagenTwoTower)
