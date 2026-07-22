"""Per-axis TOML override config.

Multi-task heads (one per prediction axis, e.g. ``direction_of_effect``,
``phenotype_category``) mostly share defaults inferred at model-build time.
The ``[axes.<name>]`` table in ``settings.toml`` lets a specific axis override
a subset of those defaults (loss kind, class weights, embedding size, ...)
without touching code.

Public API:
    >>> from src.config import get_axes_config
    >>> axes = get_axes_config()
    >>> axes.overrides.get("direction_of_effect")
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.config.settings import CONFIG_DATA_DIR, _load_toml


class AxisOverride(BaseModel):
    """Optional per-axis overrides for a single prediction target.

    Every field is optional — an axis only needs to declare the values it
    wants to override; anything left as ``None`` falls back to the model's
    inferred default.
    """

    kind: str | None = None
    pos_weight: float | None = None
    focal_gamma: float | None = None
    class_weights: list[float] | None = None
    embedding_dim: int | None = None
    enabled: bool | None = None


class AxesConfig(BaseModel):
    """Collection of per-axis overrides, keyed by axis/target name."""

    overrides: dict[str, AxisOverride] = Field(default_factory=dict)


def get_axes_config() -> AxesConfig:
    """Load the ``[axes]`` table from ``settings.toml``.

    Returns an empty :class:`AxesConfig` when the table is absent so callers
    can always treat the result as a normal (possibly empty) mapping.
    """
    settings_cfg = _load_toml(CONFIG_DATA_DIR / "settings.toml")
    axes_raw: dict[str, Any] = settings_cfg.get("axes", {})
    overrides = {name: AxisOverride(**tbl) for name, tbl in axes_raw.items()}
    return AxesConfig(overrides=overrides)
