"""Tests for src.config.settings."""

from src.config.settings import Settings, get_settings


class TestGetSettings:
    def test_returns_settings(self) -> None:
        s = get_settings()
        assert isinstance(s, Settings)

    def test_cached(self) -> None:
        a = get_settings()
        b = get_settings()
        assert a is b

    def test_paths_are_absolute(self) -> None:
        s = get_settings()
        assert s.paths.data.is_absolute()
        assert s.paths.logs.is_absolute()
        assert s.paths.ref_genome_fasta.is_absolute()

    def test_multi_label_set_view(self) -> None:
        s = get_settings()
        assert isinstance(s.multi_label_set, set)
        assert s.multi_label_set == set(s.multi_label_cols)

    def test_seed_default_or_toml(self) -> None:
        s = get_settings()
        # seed comes from settings.toml (711) but value isn't important here —
        # just that it's a positive int.
        assert isinstance(s.seed, int)
        assert s.seed > 0

    def test_project_name(self) -> None:
        s = get_settings()
        assert s.project_name == "Pharmagen"


class TestLegacyShim:
    def test_dirs_match_paths(self) -> None:
        from src.config.manager import DIRS

        s = get_settings()
        assert DIRS["data"] == s.paths.data
        assert DIRS["logs"] == s.paths.logs
        assert DIRS["models"] == s.paths.models

    def test_legacy_get_model_config_returns_dict(self) -> None:
        from src.config.manager import get_model_config

        cfg = get_model_config("TwoTowerGAT")
        assert isinstance(cfg, dict)
        assert "features" in cfg
        assert "targets" in cfg
        assert "params" in cfg
        assert "params_optuna" in cfg

    def test_legacy_constants_exposed(self) -> None:
        from src.config.manager import (
            DATA_DIR,
            METADATA,
            MULTI_LABEL_COLS,
            PROJECT_ROOT,
            REF_GENOME_DIR,
            REF_GENOME_FASTA,
            SEED,
            VERSION,
        )

        assert PROJECT_ROOT.is_absolute()
        assert DATA_DIR.is_absolute()
        assert REF_GENOME_DIR.is_absolute()
        assert REF_GENOME_FASTA.is_absolute()
        assert isinstance(SEED, int)
        assert isinstance(VERSION, str)
        assert isinstance(METADATA, dict)
        assert isinstance(MULTI_LABEL_COLS, set)
