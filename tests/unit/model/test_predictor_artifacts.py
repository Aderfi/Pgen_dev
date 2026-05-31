"""Predictor artifact loading — bundle vs. legacy format.

These tests exercise only the artifact-loading path: we monkeypatch
``_load_model`` so we don't need real graph data or a real checkpoint, and
we point ``get_settings().paths.encoders`` at a tmp dir via a fake settings
object. The goal is to catch regressions in how the predictor reads the
pickle written by ``pipeline._persist_training_artifacts``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import joblib
import pytest
from sklearn.preprocessing import LabelEncoder

from src.model.engine import predictor as predictor_module
from src.model.engine.predictor import PGenPredictor


@pytest.fixture
def fake_encoders():
    le = LabelEncoder()
    le.fit(["A", "B", "C", "__UNKNOWN__"])
    return {"phenotype_category": le}


@pytest.fixture
def patched_predictor(monkeypatch, tmp_path):
    """Redirect ``paths.encoders`` to tmp_path and stub the model loader.

    Returns the encoders directory so each test can drop its artifact in.
    """
    enc_dir = tmp_path / "encoders"
    enc_dir.mkdir()

    real_get_settings = predictor_module.get_settings

    def fake_get_settings():
        real = real_get_settings()
        fake_paths = SimpleNamespace(
            encoders=enc_dir,
            models=real.paths.models,
            library=real.paths.library,
            data=real.paths.data,
            project_root=real.paths.project_root,
        )
        return SimpleNamespace(
            paths=fake_paths,
            multi_label_set=real.multi_label_set,
        )

    monkeypatch.setattr(predictor_module, "get_settings", fake_get_settings)
    monkeypatch.setattr(PGenPredictor, "_load_model", lambda self: MagicMock())
    return enc_dir


def test_bundle_format_loads_dims(patched_predictor, fake_encoders):
    bundle = {
        "encoders": fake_encoders,
        "drug_dim": 22,
        "geno_dim": 11,
        "schema_version": 1,
    }
    joblib.dump(bundle, patched_predictor / "encoders_TwoTowerGAT.pkl")

    predictor = PGenPredictor("TwoTowerGAT")
    assert set(predictor.encoders) == {"phenotype_category"}
    assert list(predictor.encoders["phenotype_category"].classes_) == list(
        fake_encoders["phenotype_category"].classes_
    )
    assert predictor._saved_drug_dim == 22
    assert predictor._saved_geno_dim == 11
    assert predictor._resolve_tower_dim(predictor._saved_drug_dim, "drugs") == 22
    assert predictor._resolve_tower_dim(predictor._saved_geno_dim, "geno") == 11


def test_legacy_plain_dict_falls_back_to_cfg_extras(
    patched_predictor, fake_encoders, caplog
):
    joblib.dump(fake_encoders, patched_predictor / "encoders_TwoTowerGAT.pkl")

    with caplog.at_level("WARNING", logger=predictor_module.__name__):
        predictor = PGenPredictor("TwoTowerGAT")

    assert set(predictor.encoders) == {"phenotype_category"}
    assert predictor._saved_drug_dim is None
    assert predictor._saved_geno_dim is None
    assert any("legacy encoders pickle" in r.message for r in caplog.records)
    assert predictor._resolve_tower_dim(None, "drugs") == 25
    assert predictor._resolve_tower_dim(None, "geno") == 9


def test_missing_target_encoder_raises_encoder_error(patched_predictor):
    from src.core import EncoderError

    joblib.dump(
        {"encoders": {}, "drug_dim": 22, "geno_dim": 11, "schema_version": 1},
        patched_predictor / "encoders_TwoTowerGAT.pkl",
    )

    with pytest.raises(EncoderError, match="phenotype_category"):
        PGenPredictor("TwoTowerGAT")
