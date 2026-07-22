import pytest
from pydantic import ValidationError

from src.model.architectures.config import AxisSpec, PharmagenConfig


def test_axisspec_is_pydantic_and_validates():
    spec = AxisSpec(name="pheno", dim=3, kind="multiclass")
    assert spec.embedding_dim == 32
    with pytest.raises(ValidationError):
        AxisSpec(name="x", dim=-1)


def test_config_requires_axes():
    with pytest.raises(ValidationError):
        PharmagenConfig(
            drug_in_features=6, geno_in_features=5, embedding_dim=16, axes={}
        )
