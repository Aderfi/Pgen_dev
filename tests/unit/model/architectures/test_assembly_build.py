"""``PharmagenConfig.from_dims`` owns model construction from dims + axes."""

from __future__ import annotations

from src.config.models import TrainingParams
from src.data.cache import GraphDims
from src.model.architectures import PharmagenTwoTower
from src.model.architectures.config import AxisSpec, PharmagenConfig


def test_create_from_axes():
    cfg = PharmagenConfig.from_dims(
        dims=GraphDims(
            drug_edges=0, drug_global=0, drug_admet=0, geno_edges=0, geno_function=0
        ),
        drug_dim=6,
        geno_dim=5,
        axes={"pheno": AxisSpec(name="pheno", dim=3, kind="multiclass")},
        params=TrainingParams(
            embedding_dim=16,
            hidden_dim=16,
            dropout_rate=0.0,
            n_layers=2,
            heads=2,
        ),
        switches={"use_polypharmacy": False, "use_cross_attention": False},
    )
    model = PharmagenTwoTower(cfg)
    assert isinstance(model, PharmagenTwoTower)
