import pytest
import torch
from pydantic import ValidationError
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

from src.config.axes import AxesConfig, AxisOverride
from src.model.architectures.assembly import infer_axis_specs


def test_multiclass_and_binary_and_override():
    le3 = LabelEncoder().fit(["a", "b", "c"])
    le2 = LabelEncoder().fit(["no", "yes"])
    mlb = MultiLabelBinarizer().fit([["x", "y"]])
    encoders = {"pheno": le3, "assoc": le2, "meta": mlb}
    train = {
        "pheno": torch.tensor([0, 1, 2, 1]),
        "assoc": torch.tensor([0, 1, 1, 1]),  # 1 neg / 3 pos
        "meta": torch.zeros(4, 2),
    }
    overrides = AxesConfig(overrides={"pheno": AxisOverride(kind="ordinal")})
    specs = infer_axis_specs(encoders, train, {"meta"}, overrides)
    assert specs["pheno"].kind == "ordinal" and specs["pheno"].dim == 3
    assert specs["assoc"].kind == "binary" and specs["assoc"].dim == 1
    assert abs(specs["assoc"].pos_weight - (1 / 3)) < 1e-6
    assert specs["meta"].kind == "binary" and specs["meta"].dim == 2


def test_invalid_toml_override_kind_fails_fast():
    le2 = LabelEncoder().fit(["no", "yes"])
    overrides = AxesConfig(overrides={"assoc": AxisOverride(kind="ordnal")})  # typo
    with pytest.raises(ValidationError):
        infer_axis_specs(
            {"assoc": le2}, {"assoc": torch.tensor([0, 1])}, set(), overrides
        )


def test_pos_weight_none_when_degenerate():
    le_allpos = LabelEncoder().fit(["no", "yes"])
    le_allneg = LabelEncoder().fit(["no", "yes"])
    encoders = {"allpos": le_allpos, "allneg": le_allneg}
    train = {
        "allpos": torch.tensor([1, 1, 1]),  # no negatives
        "allneg": torch.tensor([0, 0, 0]),  # no positives
    }
    specs = infer_axis_specs(encoders, train, set(), AxesConfig(overrides={}))
    assert specs["allpos"].pos_weight is None
    assert specs["allneg"].pos_weight is None


def test_empty_train_targets_tolerated_at_inference():
    """At inference time there are no train labels — must not KeyError."""
    le3 = LabelEncoder().fit(["a", "b", "c"])
    le2 = LabelEncoder().fit(["no", "yes"])
    mlb = MultiLabelBinarizer().fit([["x", "y"]])
    encoders = {"pheno": le3, "assoc": le2, "meta": mlb}
    specs = infer_axis_specs(encoders, {}, {"meta"}, AxesConfig(overrides={}))
    assert specs["pheno"].kind == "multiclass" and specs["pheno"].dim == 3
    assert specs["assoc"].kind == "binary" and specs["assoc"].pos_weight is None
    assert specs["meta"].kind == "binary" and specs["meta"].dim == 2
