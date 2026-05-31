"""Per-model configuration — Pydantic-validated.

The TOML format encodes Optuna search ranges as lists::

    embedding_dim = ["categorical", 256, 512]
    n_layers      = ["int", 4, 6]
    learning_rate = ["log", 2e-4, 8e-4]
    dropout_rate  = ["float", 0.35, 0.6]

We parse those into a discriminated union of typed specs (CategoricalSpec /
IntSpec / FloatSpec / LogSpec) so the tuner can dispatch on ``.kind`` instead
of re-parsing strings every trial.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal, Union

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from pydantic import BaseModel, ConfigDict, Discriminator, Field, model_validator

from src.config.settings import CONFIG_DATA_DIR

# ---------------------------------------------------------------------------
# Optuna search-space specs
# ---------------------------------------------------------------------------


class _BaseSpec(BaseModel):
    model_config = ConfigDict(frozen=True)


class CategoricalSpec(_BaseSpec):
    kind: Literal["categorical"]
    choices: list[Any] = Field(..., min_length=1)


class IntSpec(_BaseSpec):
    kind: Literal["int"]
    low: int
    high: int

    @model_validator(mode="after")
    def _check_order(self) -> IntSpec:
        if self.high < self.low:
            msg = f"int spec: high ({self.high}) must be >= low ({self.low})"
            raise ValueError(msg)
        return self


class FloatSpec(_BaseSpec):
    kind: Literal["float"]
    low: float
    high: float

    @model_validator(mode="after")
    def _check_order(self) -> FloatSpec:
        if self.high < self.low:
            msg = f"float spec: high ({self.high}) must be >= low ({self.low})"
            raise ValueError(msg)
        return self


class LogSpec(_BaseSpec):
    """Log-uniform float distribution. Both bounds must be positive."""

    kind: Literal["log"]
    low: float
    high: float

    @model_validator(mode="after")
    def _check_positive_and_order(self) -> LogSpec:
        if self.low <= 0 or self.high <= 0:
            msg = f"log spec: bounds must be positive (got low={self.low}, high={self.high})"
            raise ValueError(msg)
        if self.high < self.low:
            msg = f"log spec: high ({self.high}) must be >= low ({self.low})"
            raise ValueError(msg)
        return self


OptunaSpec = Annotated[
    Union[CategoricalSpec, IntSpec, FloatSpec, LogSpec],
    Discriminator("kind"),
]


def _parse_optuna_value(raw: Any) -> Any:
    """Convert a TOML list like ``["log", 1e-4, 1e-2]`` into a typed spec.

    Non-list values (single ints, floats) pass through unchanged so they can
    be used as fixed hyperparameters in the same TOML section.
    """
    if not isinstance(raw, list) or not raw:
        return raw

    head = raw[0]
    if not isinstance(head, str):
        return raw

    kind = head.lower()
    rest = raw[1:]

    if kind == "categorical":
        return CategoricalSpec(kind="categorical", choices=rest)
    if kind == "int" and len(rest) == 2:
        return IntSpec(kind="int", low=int(rest[0]), high=int(rest[1]))
    if kind == "float" and len(rest) == 2:
        return FloatSpec(kind="float", low=float(rest[0]), high=float(rest[1]))
    if kind == "log" and len(rest) == 2:
        return LogSpec(kind="log", low=float(rest[0]), high=float(rest[1]))

    msg = f"unrecognized Optuna spec: {raw!r}"
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------


class ModelConfig(BaseModel):
    """Configuration for a single trained model.

    `params` and `optuna` are kept as flexible dicts because their keys depend
    on the architecture (DeepFM and TwoTowerGAT have different hyperparameters).
    `optuna` values are always typed specs.
    """

    model_config = ConfigDict(populate_by_name=True)

    name: str
    cols: list[str]
    features: list[str] = Field(..., min_length=1)
    targets: list[str] = Field(..., min_length=1)
    stratify_col: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    optuna: dict[str, OptunaSpec | Any] = Field(default_factory=dict)
    extras: dict[str, Any] = Field(
        default_factory=dict,
        description="Architecture-specific extras (e.g. drug_node_features).",
    )

    def fixed_params(self) -> dict[str, Any]:
        """Hyperparameters that are not part of the search space — for Optuna,
        these are passed through unchanged."""
        return dict(self.params)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _models_toml_path() -> Path:
    return CONFIG_DATA_DIR / "models.toml"


@lru_cache(maxsize=1)
def _load_models_toml() -> dict[str, Any]:
    path = _models_toml_path()
    if not path.exists():
        msg = f"models.toml not found at {path}"
        raise FileNotFoundError(msg)
    with path.open("rb") as f:
        return tomllib.load(f)


def get_available_models() -> list[str]:
    """Return the list of model names defined in models.toml."""
    return list(_load_models_toml().keys())


def get_model_config(name: str) -> ModelConfig:
    """Load and validate the configuration for a single model.

    The TOML structure under ``[ModelName]`` is mostly free-form architecture
    metadata, with reserved subsections ``[ModelName.params]`` (fixed hyper-
    parameters) and ``[ModelName.optuna]`` (search space).
    """
    models_cfg = _load_models_toml()
    if name not in models_cfg:
        available = ", ".join(_load_models_toml().keys()) or "(none)"
        msg = f"Model {name!r} not found in models.toml. Available: {available}"
        raise ValueError(msg)

    raw = dict(models_cfg[name])  # copy — don't mutate the cached dict

    params = raw.pop("params", {}) or {}
    optuna_raw = raw.pop("optuna", {}) or {}
    optuna_parsed = {k: _parse_optuna_value(v) for k, v in optuna_raw.items()}

    cols = raw.pop("cols", [])
    features = raw.pop("features", [])
    targets = raw.pop("targets", [])
    stratify_col = raw.pop("stratify_col", None)

    return ModelConfig(
        name=name,
        cols=cols,
        features=features,
        targets=targets,
        stratify_col=stratify_col,
        params=params,
        optuna=optuna_parsed,
        extras=raw,  # whatever's left (drug_node_features, etc.)
    )
