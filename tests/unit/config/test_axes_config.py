"""Unit tests for src.config.axes."""

from __future__ import annotations

from src.config.axes import AxesConfig, AxisOverride


def test_axes_override_parses():
    cfg = AxesConfig(overrides={"direction_of_effect": AxisOverride(kind="ordinal")})
    assert cfg.overrides["direction_of_effect"].kind == "ordinal"


def test_missing_override_is_none():
    cfg = AxesConfig(overrides={})
    assert cfg.overrides.get("phenotype_category") is None
