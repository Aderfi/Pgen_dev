"""Backward-compatible flat-constants view of the typed Settings.

Old code uses imports like::

    from src.config.manager import DIRS, SEED, MULTI_LABEL_COLS, get_model_config
    from src.config.manager import REF_GENOME_FASTA, DATA_DIR, PROJECT_ROOT

This module preserves those names by deriving them from
:func:`src.config.get_settings`. New code should import from
``src.config`` directly:

    from src.config import get_settings, get_model_config

Phase 4+ will progressively migrate callers; this shim disappears once nothing
imports from it anymore.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.config.models import (
    ModelConfig,
)
from src.config.models import (
    get_available_models as _get_available_models,
)
from src.config.models import (
    get_model_config as _get_model_config,
)
from src.config.settings import (
    PROJECT_ROOT,
    Settings,
    get_settings,
)

# Eagerly resolve the singleton so import order matches the legacy behaviour
# (callers expect DIRS/SEED to exist as soon as the module is imported).
_settings: Settings = get_settings()

# Ensure the project's mutable directories exist — old manager.py did this on
# import, so do the same here for back-compat. The new code path
# (``from src.config import get_settings``) does NOT trigger this; callers
# that want directory creation must call ``settings.paths.ensure_dirs()``.
_settings.paths.ensure_dirs()


# ---------------------------------------------------------------------------
# Legacy flat constants
# ---------------------------------------------------------------------------

SEED: int = _settings.seed
VERSION: str = _settings.version
METADATA: dict[str, Any] = {
    "project_name": _settings.project_name,
    "version": _settings.version,
    "author": _settings.author,
}
DATE_STAMP: str = datetime.now(tz=timezone.utc).strftime("%Y_%m_%d")
MULTI_LABEL_COLS: set[str] = _settings.multi_label_set

DIRS: dict[str, Any] = {
    "base": _settings.paths.project_root,
    "data": _settings.paths.data,
    "logs": _settings.paths.logs,
    "results": _settings.paths.results,
    "reports": _settings.paths.reports,
    "models": _settings.paths.models,
    "encoders": _settings.paths.encoders,
}

DATA_DIR = _settings.paths.data
REF_GENOME_DIR = _settings.paths.ref_genome_dir
REF_GENOME_FASTA = _settings.paths.ref_genome_fasta

# Legacy path used by some library builders
LIBRARY = PROJECT_ROOT / "src" / "library"


# ---------------------------------------------------------------------------
# Legacy callable API
# ---------------------------------------------------------------------------


def get_available_models() -> list[str]:
    """Legacy passthrough."""
    return _get_available_models()


def _model_config_as_dict(cfg: ModelConfig) -> dict[str, Any]:
    """Flatten a typed ModelConfig back into the dict shape that legacy callers
    expect. Old code accessed ``cfg["features"]``, ``cfg["params"]``, etc.
    """
    flat: dict[str, Any] = {
        "cols": cfg.cols,
        "features": cfg.features,
        "targets": cfg.targets,
        "stratify_col": cfg.stratify_col,
        "params": dict(cfg.params),
        "params_optuna": dict(cfg.optuna),
    }
    flat.update(cfg.extras)
    return flat


def get_model_config(model_name: str) -> dict[str, Any]:
    """Legacy dict-shaped view of the new typed ModelConfig.

    Returns the same keys as the pre-refactor manager: ``features``, ``targets``,
    ``params``, ``params_optuna``, ``cols``, plus any architecture extras.
    """
    return _model_config_as_dict(_get_model_config(model_name))
