"""Tests for src.data.encoders."""

import polars as pl
import pytest
import torch

from src.core import EncoderError
from src.data.encoders import UNKNOWN_CATEGORY_LABEL, TargetEncoder


class TestSingleLabel:
    def test_fit_and_transform(self) -> None:
        df = pl.DataFrame({"phenotype": ["Normal", "Poor", "Normal", "Rapid"]})
        enc = TargetEncoder(target_cols=["phenotype"], multilabel_cols=[])
        out = enc.fit_transform(df)
        assert out["phenotype"].dtype == torch.int64
        assert out["phenotype"].shape == (4,)
        # Two distinct samples should have the same index (row 0 and 2 are both 'Normal').
        assert out["phenotype"][0].item() == out["phenotype"][2].item()

    def test_encoder_persisted(self) -> None:
        df = pl.DataFrame({"phenotype": ["a", "b"]})
        enc = TargetEncoder(target_cols=["phenotype"], multilabel_cols=[])
        enc.fit_transform(df)
        assert "phenotype" in enc.encoders

    def test_reuse_encoders_unknown_value_falls_back(self) -> None:
        train = pl.DataFrame({"phenotype": ["a", "b"]})
        train_enc = TargetEncoder(target_cols=["phenotype"], multilabel_cols=[])
        train_enc.fit_transform(train)

        # Reuse on a val DataFrame containing an unseen class — must not crash.
        val = pl.DataFrame({"phenotype": ["a", "b", "novel_label"]})
        val_enc = TargetEncoder(
            target_cols=["phenotype"],
            multilabel_cols=[],
            encoders=train_enc.encoders,
        )
        out = val_enc.fit_transform(val)
        assert out["phenotype"].shape == (3,)


class TestMultiLabel:
    def test_fit_and_transform(self) -> None:
        df = pl.DataFrame({"adr": ["nausea|rash", "headache", "nausea"]})
        enc = TargetEncoder(target_cols=["adr"], multilabel_cols=["adr"])
        out = enc.fit_transform(df)
        assert out["adr"].dtype == torch.float32
        assert out["adr"].shape[0] == 3
        # 3 distinct labels: headache, nausea, rash.
        assert out["adr"].shape[1] == 3

    def test_unknown_value_yields_empty_row(self) -> None:
        df = pl.DataFrame({"adr": ["Unknown", "rash"]})
        enc = TargetEncoder(target_cols=["adr"], multilabel_cols=["adr"])
        out = enc.fit_transform(df)
        # Row 0 (Unknown) should be all zeros.
        assert out["adr"][0].sum().item() == 0.0
        assert out["adr"][1].sum().item() > 0.0


class TestErrors:
    def test_missing_unknown_label_raises(self) -> None:
        # If a caller passes a pre-fit LabelEncoder that lacks the
        # __UNKNOWN__ class, encoding should raise EncoderError.
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        le.fit(["a", "b"])  # no __UNKNOWN__
        enc = TargetEncoder(
            target_cols=["x"],
            multilabel_cols=[],
            encoders={"x": le},
        )
        with pytest.raises(EncoderError, match="UNKNOWN"):
            enc.fit_transform(pl.DataFrame({"x": ["a", "b"]}))

    def test_unknown_constant(self) -> None:
        assert UNKNOWN_CATEGORY_LABEL == "__UNKNOWN__"
