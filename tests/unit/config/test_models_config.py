"""Tests for the typed model configuration loader (src.config.models)."""

import pytest

from src.config.models import (
    CategoricalSpec,
    FloatSpec,
    IntSpec,
    LogSpec,
    ModelConfig,
    _parse_optuna_value,
    get_available_models,
    get_model_config,
)


class TestParseOptunaValue:
    def test_categorical(self) -> None:
        spec = _parse_optuna_value(["categorical", 256, 512])
        assert isinstance(spec, CategoricalSpec)
        assert spec.choices == [256, 512]

    def test_int(self) -> None:
        spec = _parse_optuna_value(["int", 4, 6])
        assert isinstance(spec, IntSpec)
        assert spec.low == 4
        assert spec.high == 6

    def test_float(self) -> None:
        spec = _parse_optuna_value(["float", 0.1, 0.5])
        assert isinstance(spec, FloatSpec)
        assert spec.low == 0.1
        assert spec.high == 0.5

    def test_log(self) -> None:
        spec = _parse_optuna_value(["log", 1e-4, 1e-2])
        assert isinstance(spec, LogSpec)
        assert spec.low == pytest.approx(1e-4)
        assert spec.high == pytest.approx(1e-2)

    def test_int_high_below_low_rejected(self) -> None:
        with pytest.raises(ValueError, match="high"):
            _parse_optuna_value(["int", 10, 5])

    def test_log_with_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            _parse_optuna_value(["log", 0.0, 1.0])

    def test_log_with_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            _parse_optuna_value(["log", -1.0, 1.0])

    def test_unknown_kind_rejected(self) -> None:
        with pytest.raises(ValueError, match="unrecognized"):
            _parse_optuna_value(["uniform", 0.0, 1.0])

    def test_scalar_passes_through(self) -> None:
        assert _parse_optuna_value(0.1) == 0.1
        assert _parse_optuna_value(42) == 42
        assert _parse_optuna_value("static") == "static"

    def test_empty_list_passes_through(self) -> None:
        assert _parse_optuna_value([]) == []

    def test_list_with_non_string_head_passes_through(self) -> None:
        assert _parse_optuna_value([1, 2, 3]) == [1, 2, 3]


class TestGetModelConfig:
    def test_known_model_loads(self) -> None:
        cfg = get_model_config("TwoTowerGAT")
        assert isinstance(cfg, ModelConfig)
        assert cfg.name == "TwoTowerGAT"
        assert "drugs_cid" in cfg.features
        assert "phenotype_category" in cfg.targets

    def test_unknown_model_raises_with_available_list(self) -> None:
        with pytest.raises(ValueError, match="not found in models.toml"):
            get_model_config("NotAModel")

    def test_optuna_specs_are_typed(self) -> None:
        cfg = get_model_config("TwoTowerGAT")
        assert isinstance(cfg.optuna["embedding_dim"], CategoricalSpec)
        assert isinstance(cfg.optuna["n_layers"], IntSpec)
        assert isinstance(cfg.optuna["dropout_rate"], FloatSpec)
        assert isinstance(cfg.optuna["learning_rate"], LogSpec)

    def test_extras_capture_unstructured_keys(self) -> None:
        cfg = get_model_config("TwoTowerGAT")
        assert "drug_node_features" in cfg.extras
        assert "geno_node_features" in cfg.extras

    def test_get_available_models(self) -> None:
        models = get_available_models()
        assert "TwoTowerGAT" in models
        assert all(isinstance(m, str) for m in models)
