"""Target-encoding utilities extracted from ``DoubleTowerDataset``.

The dataset's ``_encode_targets`` / ``_encode_singlelabel`` /
``_encode_multilabel`` methods became hard to test because they were tied
to the dataset constructor. Pulling them into a focused
``TargetEncoder`` class lets you fit on a training DataFrame, persist the
encoders, and reuse them at validation / inference time without
constructing a full Dataset.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

import polars as pl
import torch
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

from src.core import EncoderError

logger = logging.getLogger(__name__)


UNKNOWN_CATEGORY_LABEL = "__UNKNOWN__"


class TargetEncoder:
    """Fit/transform encoders for single-label and multi-label target columns.

    Re-uses encoders across train/val/test by passing them via ``encoders=``
    at construction. New columns trigger fit; existing encoders trigger
    transform-only.
    """

    def __init__(
        self,
        target_cols: Iterable[str],
        multilabel_cols: Iterable[str],
        *,
        encoders: dict[str, Any] | None = None,
    ) -> None:
        self.target_cols: list[str] = list(target_cols)
        self.multilabel_cols: set[str] = set(multilabel_cols)
        self.encoders: dict[str, Any] = encoders or {}

    # ----- public API ------------------------------------------------------ #

    def fit_transform(self, df: pl.DataFrame) -> dict[str, torch.Tensor]:
        """Encode every target column, fitting on first sight."""
        encoded: dict[str, torch.Tensor] = {}
        for col in self.target_cols:
            try:
                series = self._stringify(df, col)
                if col in self.multilabel_cols:
                    encoded[col] = self._encode_multilabel(col, series)
                else:
                    encoded[col] = self._encode_singlelabel(col, series)
            except Exception as e:
                msg = f"Failed to encode target {col!r}: {e}"
                raise EncoderError(msg) from e
        logger.debug("Encoded %d targets.", len(encoded))
        return encoded

    # ----- internals ------------------------------------------------------- #

    @staticmethod
    def _stringify(df: pl.DataFrame, col: str) -> pl.Series:
        return df.select(pl.col(col).cast(pl.String).fill_null("")).to_series()

    def _encode_multilabel(self, col: str, series: pl.Series) -> torch.Tensor:
        """Multi-label → float tensor of shape ``[N, num_classes]``."""
        parsed = (
            series.to_frame()
            .select(
                pl.when(pl.col(col) != "Unknown")
                .then(pl.col(col).str.split("|"))
                .otherwise(pl.lit([], dtype=pl.List(pl.String)))
            )
            .to_series()
            .to_list()
        )
        if col in self.encoders:
            mlb: MultiLabelBinarizer = self.encoders[col]
            matrix = mlb.transform(parsed)
        else:
            mlb = MultiLabelBinarizer()
            matrix = mlb.fit_transform(parsed)
            self.encoders[col] = mlb
            logger.debug(
                "Fitted MultiLabelBinarizer for %s: %d classes", col, len(mlb.classes_)
            )
        return torch.tensor(matrix, dtype=torch.float32)

    def _encode_singlelabel(self, col: str, series: pl.Series) -> torch.Tensor:
        """Single-label → long tensor of shape ``[N]``."""
        if col not in self.encoders:
            uniques = series.unique().to_list()
            unique_values = sorted({*uniques, UNKNOWN_CATEGORY_LABEL})
            le = LabelEncoder()
            le.fit(unique_values)
            self.encoders[col] = le
            logger.debug(
                "Fitted LabelEncoder for %s: %d classes", col, len(le.classes_)
            )

        le = self.encoders[col]
        mapping = {label: idx for idx, label in enumerate(le.classes_)}
        unknown_idx = mapping.get(UNKNOWN_CATEGORY_LABEL)
        if unknown_idx is None:
            msg = (
                f"target {col!r} encoder has no {UNKNOWN_CATEGORY_LABEL!r} class — "
                "re-fit required."
            )
            raise EncoderError(msg)

        indices = series.replace(mapping, default=unknown_idx)
        # ``.to_numpy()`` on a Polars Int64 series may return a read-only view;
        # copy so torch can wrap it without a (loud) warning.
        return torch.from_numpy(indices.cast(pl.Int64).to_numpy().copy()).long()
