"""Runtime validation utilities for Pharmagen.

Provides lightweight guards for data quality and configuration sanity that
complement Pydantic's schema validation at system boundaries.

Classes:
    :class:`ConfigValidator` — column presence and Optuna search-space checks.
    :class:`DataValidator`   — missing-value and class-balance inspection.
"""

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)


class ConfigValidator:
    """Checks that configuration objects satisfy runtime constraints."""

    @staticmethod
    def validate_data_columns(
        df_columns: Sequence[str],
        required_features: Sequence[str],
        required_targets: Sequence[str],
    ) -> bool:
        """Raise ``ValueError`` if the DataFrame is missing required columns."""
        missing_features = [f for f in required_features if f not in df_columns]
        missing_targets = [t for t in required_targets if t not in df_columns]

        if missing_features:
            raise ValueError(f"DataFrame missing required feature columns: {missing_features}")
        if missing_targets:
            raise ValueError(f"DataFrame missing required target columns: {missing_targets}")

        return True

    @staticmethod
    def validate_optuna_params(params: Mapping[str, Any]) -> bool:
        """Return ``True`` if all Optuna search-space entries look sane.

        Accepts both raw list specs ``["int", lo, hi]`` (legacy) and typed
        :class:`~src.config.models.OptunaSpec` objects (current).
        """
        for param_name, spec in params.items():
            if param_name in {"patience", "epochs"}:
                continue
            # Typed OptunaSpec objects already passed Pydantic validation.
            if hasattr(spec, "kind"):
                continue
            # Legacy list format — do a minimal sanity check.
            if not isinstance(spec, list) or not spec:
                logger.warning("Optuna param %r is not a list spec; skipping.", param_name)
                continue
            ptype = spec[0]
            if ptype in {"int", "float", "log"} and len(spec) >= 3:
                if spec[1] >= spec[2]:
                    logger.error(
                        "Optuna param %r has min (%s) >= max (%s).",
                        param_name, spec[1], spec[2],
                    )
                    return False
            elif ptype == "categorical" and len(spec) < 2:
                logger.error("Optuna categorical param %r needs at least one choice.", param_name)
                return False

        return True


class DataValidator:
    """Inspects DataFrame quality at runtime."""

    @staticmethod
    def check_missing_values(
        df: pl.DataFrame,
        columns: Sequence[str],
        threshold: float = 0.5,
    ) -> dict[str, float]:
        """Return per-column missing-value fractions; warn if any exceed *threshold*."""
        valid_cols = [c for c in columns if c in df.columns]
        for c in set(columns) - set(valid_cols):
            logger.warning("Column %r not found in DataFrame — skipping missing-value check.", c)

        if not valid_cols:
            return {}

        stats = df.select(
            pl.col(c).is_null().mean().alias(c) for c in valid_cols
        ).row(0, named=True)

        for col, frac in stats.items():
            if frac > threshold:
                logger.warning(
                    "Column %r has %.1f%% missing values (threshold: %.1f%%).",
                    col, frac * 100, threshold * 100,
                )

        return dict(stats)

    @staticmethod
    def check_class_balance(
        df: pl.DataFrame,
        target_column: str,
        min_samples_per_class: int = 10,
    ) -> dict[str, int]:
        """Return class counts; warn about rare classes below *min_samples_per_class*."""
        if target_column not in df.columns:
            logger.error("Target column %r not found.", target_column)
            return {}

        counts = {row[0]: row[1] for row in df[target_column].value_counts().iter_rows()}
        rare = {cls: n for cls, n in counts.items() if n < min_samples_per_class}
        if rare:
            logger.warning(
                "Target %r has rare classes (< %d samples): %s",
                target_column, min_samples_per_class, rare,
            )

        return counts
