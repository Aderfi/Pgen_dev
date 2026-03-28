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

"""Configuration validation utilities for Pharmagen.

This module provides validation functions for configuration files
to catch errors early and provide helpful error messages.
"""

import logging
from collections.abc import Mapping, MutableMapping, MutableSequence, Sequence
from pathlib import Path
from typing import Any, cast

import polars as pl

from src.utils.exceptions import ValidationError

logger = logging.getLogger(__name__)


class ConfigValidator:
    """Validates configuration files and settings.

    Follows SRP: only responsible for configuration validation.
    """

    @staticmethod
    def validate_model_config(config: Mapping[str, object], model_name: str) -> bool:
        """Validate a model configuration dictionary.

        Args:
            config: Model configuration dictionary.
            model_name: Name of the model for error messages.

        Returns:
            True if valid, False otherwise.

        Raises:
            ValueError: If critical validation fails.
        """
        required_keys = ["features", "targets", "params"]
        missing_keys = [k for k in required_keys if k not in config]

        if missing_keys:
            raise ValueError(
                f"Model '{model_name}' configuration missing required keys: {missing_keys}"
            )

        # Validate features
        if not isinstance(config["features"], list) or len(config["features"]) < 1:
            raise ValueError(
                f"Model '{model_name}' must have at least one feature column"
            )

        # Validate targets
        if not isinstance(config["targets"], list) or len(config["targets"]) < 1:
            raise ValueError(
                f"Model '{model_name}' must have at least one target column"
            )

        # Validate params
        if not isinstance(config["params"], dict):
            raise ValueError(
                f"Model '{model_name}' params must be a dictionary"
            )

        # Validate critical hyperparameters
        params = config["params"]
        if "learning_rate" in params:
            lr = params["learning_rate"]
            if not isinstance(lr, (int, float)) or lr <= 0 or lr > 1:
                logger.warning(
                    f"Model '{model_name}' has unusual learning_rate: {lr}. "
                    "Expected value in (0, 1]"
                )

        if "batch_size" in params:
            bs = params["batch_size"]
            if not isinstance(bs, int) or bs < 1 or bs > 1024:  # noqa
                logger.warning(
                    f"Model '{model_name}' has unusual batch_size: {bs}. "
                    "Expected value in [1, 1024]"
                )

        logger.debug(f"Configuration for '{model_name}' validated successfully")
        return True

    @staticmethod
    def validate_paths_config(paths: Mapping[str, Any], create_missing: bool = True) -> bool:
        """Validate paths configuration.

        Args:
            paths: Dictionary of path configurations.
            create_missing: If True, create missing directories.

        Returns:
            True if all paths are valid.
        """
        for path_name, path_value in paths.items():
            if not isinstance(path_value, (str, Path)):
                logger.warning(f"Path '{path_name}' has invalid type: {type(path_value)}")
                continue

            path_obj = Path(path_value)

            if not path_obj.exists():
                if create_missing:
                    try:
                        path_obj.mkdir(parents=True, exist_ok=True)
                        logger.info(f"Created directory: {path_obj}")
                    except Exception as e:
                        logger.error(f"Failed to create directory {path_obj}: {e}")
                        return False
                else:
                    logger.warning(f"Path does not exist: {path_obj}")

        return True

    @staticmethod
    def validate_optuna_params(params: Mapping[str, Any]) -> bool:
        """Validate Optuna hyperparameter search space.

        Args:
            params: Dictionary of Optuna parameter definitions.

        Returns:
            True if valid, False otherwise.
        """
        valid_types = {"categorical", "int", "float", "log"}

        for param_name, param_def in params.items():
            if param_name in {"patience", "epochs"}:
                # These are configuration, not search params
                continue

            if not isinstance(param_def, list) or len(param_def) == 0:
                logger.warning(
                    f"Optuna param '{param_name}' should be a list, got {type(param_def)}"
                )
                continue

            param_type = param_def[0]

            if param_type not in valid_types:
                logger.warning(
                    f"Optuna param '{param_name}' has invalid type '{param_type}'. "
                    f"Expected one of: {valid_types}"
                )
                continue

            # Validate ranges
            if param_type in {"int", "float", "log"}:
                if len(param_def) < 3: # noqa
                    logger.warning(
                        f"Optuna param '{param_name}' type '{param_type}' "
                        "requires [type, min, max]"
                    )
                    continue

                min_val, max_val = param_def[1], param_def[2]
                if min_val >= max_val:
                    logger.error(
                        f"Optuna param '{param_name}' has min ({min_val}) >= max ({max_val})"
                    )
                    return False

            elif param_type == "categorical":
                if len(param_def) < 2: # noqa
                    logger.error(
                        f"Optuna categorical param '{param_name}' needs at least one choice"
                    )
                    return False

        logger.debug("Optuna parameters validated successfully")
        return True

    @staticmethod
    def validate_data_columns(
        df_columns: MutableSequence[str],
        required_features: MutableSequence[str],
        required_targets: MutableSequence[str]
    ) -> bool:
        """Validate that DataFrame has required columns.

        Args:
            df_columns: List of column names in DataFrame.
            required_features: Required feature columns.
            required_targets: Required target columns.

        Returns:
            True if valid, False otherwise.

        Raises:
            ValueError: If critical columns are missing.
        """
        missing_features = [f for f in required_features if f not in df_columns]
        missing_targets = [t for t in required_targets if t not in df_columns]

        if missing_features:
            raise ValueError(
                f"DataFrame missing required feature columns: {missing_features}"
            )

        if missing_targets:
            raise ValueError(
                f"DataFrame missing required target columns: {missing_targets}"
            )

        return True


class DataValidator:
    """Validates data quality and consistency.

    Follows SRP: only responsible for data validation.
    """

    @staticmethod
    def check_missing_values(
        df: pl.DataFrame,
        columns: MutableSequence[str],
        threshold: float = 0.5
    ) -> Mapping[str, float]:
        """Check for missing values in specified columns.

        Args:
            df: DataFrame to check.
            columns: List of column names to check.
            threshold: Maximum allowed fraction of missing values.

        Returns:
            Dictionary mapping column names to fraction of missing values.
        """
        missing_stats = {}
        valid_cols = [c for c in columns if c in df.columns]
        missing_cols_input = set(columns) - set(valid_cols)

        if missing_cols_input:
             for c in missing_cols_input:
                 logger.warning(f"Column '{c}' not found in DataFrame")

        if not valid_cols:
            return {}
        try:
            stats_df = df.select([
                pl.col(c).is_null().mean().alias(c) for c in valid_cols
            ])

            stats_dict = stats_df.row(0, named=True)

            # 3. Verificación de umbrales y logging
            for col, missing_fraction in stats_dict.items():
                missing_stats[col] = missing_fraction

                if missing_fraction > threshold:
                    logger.warning(
                        f"Column '{col}' has {missing_fraction:.1%} missing values "
                        f"(threshold: {threshold:.1%})"
                    )
        except ValidationError as e:
            logger.error(f"Error checking missing values: {e}")
            return {}

        return missing_stats

    @staticmethod
    def check_class_balance(
        df: pl.DataFrame,
        target_column: str,
        min_samples_per_class: int = 10
    ) -> Mapping[str, int]:
        """Check class balance for a target column.

        Args:
            df: DataFrame to check.
            target_column: Name of target column.
            min_samples_per_class: Minimum samples required per class.

        Returns:
            Dictionary mapping class labels to counts.
        """
        if target_column not in df.columns:
            logger.error(f"Target column '{target_column}' not found")
            return {}

        vc_df = df[target_column].value_counts()
        class_counts = {row[0]: row[1] for row in vc_df.iter_rows()}

        rare_classes = {
            cls: count for cls, count in class_counts.items()
                if count < min_samples_per_class
        }

        if rare_classes:
            logger.warning(
                f"Target '{target_column}' has rare classes (< {min_samples_per_class} samples): "
                f"{rare_classes}"
            )

        return class_counts

    @staticmethod
    def check_data_types(
        df: pl.DataFrame,
        expected_types: Mapping[str, type]
    ) -> bool:
        """Check that columns have expected data types.

        Args:
            df: DataFrame to check.
            expected_types: Dictionary mapping column names to expected types.

        Returns:
            True if all types match, False otherwise.
        """
        all_valid = True
        schema = df.schema

        for col, expected_py_type in expected_types.items():
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found in DataFrame")
                all_valid = False
                continue

            actual_pl_type = schema[col]
            is_col_valid = False
            type_msg = "__UNK__"

            # Check if types are compatible
            if expected_py_type is str:
                is_col_valid = actual_pl_type in (pl.String, pl.Categorical, pl.Enum, pl.Utf8)
                type_msg = "String/Categorical/Enum"

            elif expected_py_type is int:
                is_col_valid = actual_pl_type.is_integer()
                type_msg = "Integer"

            elif expected_py_type is float:
                is_col_valid = actual_pl_type.is_float()
                type_msg = "Float"

            elif expected_py_type is bool:
                is_col_valid = actual_pl_type == pl.Boolean
                type_msg = "Boolean"

            elif expected_py_type is list:
                is_col_valid = isinstance(actual_pl_type, pl.List)
                type_msg = "List"

            else:
                logger.warning(f"Type check not implemented for python type: {expected_py_type}")
                continue

            if not is_col_valid:
                logger.warning(
                    f"Column '{col}' expected {type_msg} (mapped from {expected_py_type.__name__}), "
                    f"got Polars type: {actual_pl_type}"
                )
                all_valid = False

        return all_valid

class GraphValidator:
    """Validates graph data structures."""

    @staticmethod
    def validate_graph_data(data, data_type: str) -> bool:
        """Validate PyG Data object.

        Args:
            data: torch_geometric.data.Data object
            data_type: Type of graph ("drug" or "geno")

        Returns:
            True if valid

        Raises:
            ValueError: If graph is invalid
        """
        if not hasattr(data, 'x') or not hasattr(data, 'edge_index'):
             raise ValueError(f"{data_type} graph missing fields")
        if data.x.size(0) == 0:
             raise ValueError(f"{data_type} graph has no nodes")

        if not logger.isEnabledFor(logging.DEBUG):
            pass
        return True

