import torch
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
