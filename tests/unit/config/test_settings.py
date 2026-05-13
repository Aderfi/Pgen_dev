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


