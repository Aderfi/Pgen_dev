# Pharmagen - Pharmacogenetic Prediction and Therapeutic Efficacy
# Copyright (C) 2025 Adrim Hamed Outmani
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Pharmagen - Pharmacogenetic Prediction and Therapeutic Efficacy
# Copyright (C) 2025 Adrim Hamed Outmani

"""Configuration manager for Pharmagen project.

This module provides centralized configuration management including:

- Project paths and directory structure
- Model configuration loading and merging
- Global settings and constants
- TOML configuration file parsing

Follows SOLID principles with validation and error handling.
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from src.utils.validation import ConfigValidator

logger = logging.getLogger(__name__)

# =============================================================================
# 1. CONSTANTS & PATHS
# =============================================================================

SEED = 711
_OPTUNA_RANGE_LENGTH = 2

# Root is 3 levels up from this file (src/cfg/manager.py -> src/cfg -> src -> root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(__file__).parent
LIBRARY = Path(PROJECT_ROOT / "src" / "library")

# Load Core Config Files
try:
    with (CONFIG_DIR / "paths.toml").open("rb") as f:
        _PATHS_CFG = tomllib.load(f)
    with (CONFIG_DIR / "settings.toml").open("rb") as f:
        _GLOBAL_CFG = tomllib.load(f)
    with (CONFIG_DIR / "models.toml").open("rb") as f:
        _MODELS_CFG = tomllib.load(f)
except FileNotFoundError as e:
    sys.exit(f"CRITICAL: Missing configuration file: {e}")


def _resolve(path_str: str) -> Path:
    """Resolve paths relative to PROJECT_ROOT."""
    return PROJECT_ROOT / path_str


# Exported Constants (Flat structure for easy import)
METADATA = _GLOBAL_CFG.get("metadata", {})
PROJECT_NAME = METADATA.get("project_name", "Pharmagen")
VERSION = METADATA.get("version", "0.0.0")
DATE_STAMP = datetime.now(tz=timezone.utc).strftime("%Y_%m_%d")

# Directory Map
DIRS = {
    "base": PROJECT_ROOT,  # Added explicit base
    "data": _resolve(_PATHS_CFG["base"]["data"]),
    "logs": _resolve(_PATHS_CFG["base"]["logs"]),
    "results": _resolve(_PATHS_CFG["base"]["results"]),
    "reports": _resolve(_PATHS_CFG["base"]["reports"]),
    "models": _resolve(_PATHS_CFG["models"]["models_saved"]),
    "encoders": _resolve(_PATHS_CFG["models"]["encoders"]),
}

# Ensure directories exist
for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

# Validate paths configuration
ConfigValidator.validate_paths_config(
    {k: str(v) for k, v in DIRS.items()},
    create_missing=True
)

# Helpers
MULTI_LABEL_COLS = set(_GLOBAL_CFG.get("project", {}).get("multi_label_cols", []))

# =============================================================================
# 2. MODEL CONFIGURATION LOGIC
# =============================================================================


def get_available_models() -> list[str]:
    """Return list of available model names from configuration."""
    return list(_MODELS_CFG.keys())


def _parse_optuna_param(val: list | tuple | float | str,    # deprecated?
    ) -> list | tuple | float | str:
    """Parse TOML lists into Python tuples/types for Optuna.

    Converts list ranges to tuples and preserves type-specific definitions.
    [min, max] -> (min, max)
    ["int", min, max] -> ["int", min, max] (kept as list for specific handling)

    Parameters
    ----------
    val : list | tuple | int | float | str
        Value to parse from TOML configuration.

    Returns
    -------
    list | tuple | int | float | str
        Parsed value suitable for Optuna parameter definition.

    """
    if isinstance(val, list):
        # Check for explicit type definition or range
        if len(val) > 0 and val[0] == "int":
            return val
        if (len(val) == _OPTUNA_RANGE_LENGTH
                and all(isinstance(x, (int, float)) for x in val)):
            return tuple(val)
    return val


def get_model_config(model_name: str) -> dict[str, Any]:
    """Return a merged configuration dictionary for a specific model.

    Merges global defaults with model-specific configuration.
    Priority: Model Config > Global Defaults.

    Parameters
    ----------
    model_name : str
        Name of the model to retrieve configuration for.

    Returns
    -------
    dict[str, Any]
        Complete configuration dictionary for the model.

    Raises
    ------
    ValueError
        If model is not found or missing required configuration keys.

    """
    final_config: dict[str, Any] = {}
    if model_name not in _MODELS_CFG:
        msg = f"Model '{model_name}' not found in models.toml"
        raise ValueError(msg)

    # 1. Start with defaults
    final_config = _GLOBAL_CFG.copy()
    final_config.update(final_config.pop("project", {}))

    # 2. Update with specific model config
    model_data = _MODELS_CFG[model_name].copy()

    # 3. Update params if present
    if "params" in model_data:
        final_config["params"].update(model_data.pop("params").items())

    # Process Optuna params if present
    if "optuna" in model_data:
        final_config["params_optuna"] = {
            k: v for k, v in model_data.pop("optuna").items()
        }

    final_config.update(model_data)

    # 3. Validation
    required_keys = ["features", "targets"]
    if not all(k in final_config for k in required_keys):
        msg = f"Model config requires {required_keys}"
        raise ValueError(msg)
    
    # Validate the complete configuration
    ConfigValidator.validate_model_config(final_config, model_name)
    
    # Validate Optuna params if present
    if "params_optuna" in final_config:
        ConfigValidator.validate_optuna_params(final_config["params_optuna"])

    return final_config


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logger.info("Pharmagen Config Manager v%s", VERSION)
    logger.info("Root: %s", PROJECT_ROOT)
    logger.info("Available Models: %s", get_available_models())
    model_choice = get_available_models()[0]
    logger.info("Sample Config for '%s':", model_choice)
    logger.debug("=" * 40)
    logger.debug("=" * 40)
    config = get_model_config(model_choice)
    logger.debug(config)
    logger.debug("=" * 40)
    logger.debug("\n\n\tOptuna Params: \n %s", config["params"])  # For quick inspection
