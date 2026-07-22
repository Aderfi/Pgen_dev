"""Public API for project configuration.

Prefer this entry point in new code:

    from src.config import get_settings, get_model_config, ModelConfig

The legacy ``src.config.manager`` module is a backward-compat shim that
re-exports the same data as flat constants (DIRS, SEED, …). New code should
not depend on it.
"""

from src.config.axes import AxesConfig, AxisOverride, get_axes_config
from src.config.models import (
    CategoricalSpec,
    FloatSpec,
    IntSpec,
    LogSpec,
    ModelConfig,
    OptunaSpec,
    get_available_models,
    get_model_config,
)
from src.config.paths import Paths
from src.config.settings import (
    PROJECT_ROOT,
    Settings,
    configure_logging_level,
    get_settings,
)

__all__ = [
    "AxesConfig",
    "AxisOverride",
    "CategoricalSpec",
    "FloatSpec",
    "IntSpec",
    "LogSpec",
    "ModelConfig",
    "OptunaSpec",
    "PROJECT_ROOT",
    "Paths",
    "Settings",
    "configure_logging_level",
    "get_available_models",
    "get_axes_config",
    "get_model_config",
    "get_settings",
]
