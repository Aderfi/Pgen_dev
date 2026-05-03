"""PyTorch Datasets for Pharmagen.

This module contains three datasets:

- :class:`PGenDataset`        — a tabular dataset that pre-converts every
                                column to a NumPy array for fast index
                                access (kept for the legacy DeepFM path).
- :class:`PGenProcessor`      — sklearn-style fit/transform encoder used
                                by ``PGenDataset``.
- :class:`DoubleTowerDataset` — the active two-tower GNN dataset. It now
                                composes ``GraphCache`` (in-RAM graph
                                store) and ``TargetEncoder`` (target
                                tensorization) instead of doing both
                                inline.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, MutableSequence, Set
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from polars.dataframe import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from torch.utils.data import Dataset
from torch_geometric.data.data import Data

from src.config.manager import LIBRARY, MULTI_LABEL_COLS
from src.data.cache import GraphCache, GraphDims
from src.data.encoders import UNKNOWN_CATEGORY_LABEL, TargetEncoder
from src.data.graph_indexing import GraphIndexBuilder
from src.utils.exceptions import DataError, EncoderError

logger = logging.getLogger(__name__)


# Threshold above which RAM preloading is suspicious (warn, don't refuse).
PRELOAD_THRESHOLD = 5000

# Default tower dimensions when callers don't override.
DEFAULT_DIMENSIONS: dict[str, dict[str, int]] = {
    "drugs": {"features": 25, "edges": 7, "attrs": 0},
    "geno":  {"features": 9,  "edges": 3, "attrs": 0},
}


# --------------------------------------------------------------------------- #
# PGenProcessor — sklearn-style encoder used by the legacy tabular path.
# --------------------------------------------------------------------------- #


class PGenProcessor(BaseEstimator, TransformerMixin):
    """Wraps LabelEncoder + MultiLabelBinarizer for all configured columns.

    Used by ``PGenDataset`` (tabular DeepFM). For the GNN path see
    ``TargetEncoder`` in ``src.data.encoders``.
    """

    def __init__(
        self,
        feature_cols: MutableSequence[str],
        target_cols: MutableSequence[str],
        multi_label_cols: MutableSequence[str],
    ) -> None:
        self.feature_cols = [c.lower() for c in feature_cols]
        self.target_cols = [c.lower() for c in target_cols]
        self.multi_label_cols = {c.lower() for c in multi_label_cols}
        self.encoders: dict[str, Any] = {}
        self.cols_to_process = set(self.feature_cols + self.target_cols)

    def fit(self, df: DataFrame, y: None = None) -> PGenProcessor:  # noqa: ARG002
        logger.info("Fitting encoders ...")
        for col in self.cols_to_process:
            if col not in df.columns:
                logger.warning("Column %r not found in DataFrame", col)
                continue
            if col in self.multi_label_cols:
                parsed = df.select(
                    pl.col(col).str.split("|").fill_null(
                        pl.lit([], dtype=pl.List(pl.String))
                    )
                ).to_series()
                enc = MultiLabelBinarizer()
                enc.fit(parsed)
            else:
                uniques = df.select(pl.col(col).drop_nulls().unique()).to_series()
                values = sorted({*uniques.to_list(), UNKNOWN_CATEGORY_LABEL})
                enc = LabelEncoder()
                enc.fit(values)
            self.encoders[col] = enc
        logger.info("Fitted %d encoders.", len(self.encoders))
        return self

    def transform(self, df: DataFrame) -> DataFrame:
        if not self.encoders:
            raise EncoderError("Processor not fitted. Call fit() first.")

        expressions = []
        for col, enc in self.encoders.items():
            if col not in df.columns:
                continue
            if isinstance(enc, MultiLabelBinarizer):
                def _apply(series: pl.Series, _enc: MultiLabelBinarizer = enc) -> pl.Series:
                    parsed = (
                        series.str.split("|")
                        .fill_null(pl.lit([], dtype=pl.List(pl.String)))
                        .to_list()
                    )
                    matrix = _enc.transform(parsed)
                    return pl.Series([list(row) for row in matrix], dtype=pl.List(pl.Int8))

                expressions.append(
                    pl.col(col).map_batches(_apply, return_dtype=pl.List(pl.Int8)).alias(col)
                )
            else:
                mapping = {label: idx for idx, label in enumerate(enc.classes_)}
                unknown_idx = mapping.get(UNKNOWN_CATEGORY_LABEL, -1)
                expressions.append(
                    pl.col(col).cast(pl.String)
                    .replace(mapping, default=unknown_idx)
                    .cast(pl.Int32).alias(col)
                )
        return df.with_columns(expressions)


# --------------------------------------------------------------------------- #
# PGenDataset — flat tabular dataset (legacy DeepFM path).
# --------------------------------------------------------------------------- #


class PGenDataset(Dataset):
    """Tabular dataset with contiguous NumPy backing arrays for fast access.

    Splits scalar features (LongTensor) and dense/multi-hot features
    (FloatTensor) so ``__getitem__`` is two array reads + a tensor wrap.
    """

    def __init__(
        self,
        df: DataFrame,
        feature_cols: MutableSequence[str],
        target_cols: MutableSequence[str],
        multi_label_cols: Set[str],
    ) -> None:
        self.scalar_data: dict[str, np.ndarray] = {}
        self.dense_data: dict[str, np.ndarray] = {}
        self.length = len(df)

        ml_lower = {c.lower() for c in multi_label_cols}
        for col_name in (*feature_cols, *target_cols):
            if col_name not in df.columns:
                logger.warning("Column %r not found; skipping.", col_name)
                continue
            series = df[col_name]
            key = col_name.lower()

            if key in ml_lower:
                matrix = np.array(series.to_list(), dtype=np.float32)
                self.dense_data[key] = np.ascontiguousarray(matrix)
            else:
                if series.null_count() > 0:
                    logger.warning("Nulls in scalar column %r — filling with 0.", col_name)
                    arr = series.fill_null(0).to_numpy()
                else:
                    arr = series.to_numpy()
                self.scalar_data[key] = np.ascontiguousarray(arr.astype(np.int64))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        batch: dict[str, torch.Tensor] = {}
        for col, data in self.dense_data.items():
            batch[col] = torch.from_numpy(data[idx])
        for col, data in self.scalar_data.items():
            batch[col] = torch.tensor(data[idx], dtype=torch.long)
        return batch


# --------------------------------------------------------------------------- #
# DoubleTowerDataset — active GNN dataset.
# --------------------------------------------------------------------------- #


def _dims_from_input(input_dimensions: dict[str, dict[str, int]] | None) -> GraphDims:
    """Translate the legacy nested-dict dim spec into a typed GraphDims."""
    if not input_dimensions:
        return GraphDims()

    drugs = input_dimensions.get("drugs", {})
    geno = input_dimensions.get("geno", {})
    return GraphDims(
        drug_features=drugs.get("features", DEFAULT_DIMENSIONS["drugs"]["features"]),
        drug_edges=drugs.get("edges", DEFAULT_DIMENSIONS["drugs"]["edges"]),
        geno_features=geno.get("features", DEFAULT_DIMENSIONS["geno"]["features"]),
        geno_edges=geno.get("edges", DEFAULT_DIMENSIONS["geno"]["edges"]),
    )


def _validate_input_dims(dims: dict[str, dict[str, int]]) -> None:
    required_subkeys = ("features", "edges", "attrs")
    for kind in ("drugs", "geno"):
        if kind not in dims:
            continue
        if not isinstance(dims[kind], dict):
            msg = f"dimension {kind!r} must be a dict, got {type(dims[kind]).__name__}"
            raise DataError(msg)
        for subkey in required_subkeys:
            if subkey not in dims[kind]:
                continue
            v = dims[kind][subkey]
            if not isinstance(v, int) or v < 0:
                msg = f"invalid dimension {kind}.{subkey}: {v} (must be non-negative int)"
                raise DataError(msg)


class DoubleTowerDataset(Dataset):
    """Drug-graph + genotype-graph pair dataset for the Two-Tower GNN.

    Composition:
        - ``GraphIndexBuilder``  — discovers the on-disk library.
        - ``GraphCache``         — owns the in-memory cache + dummy fallback.
        - ``TargetEncoder``      — fits/applies sklearn encoders to target cols.

    The dataset itself just glues these together and exposes ``__getitem__``.

    Args:
        df: Polars DataFrame with the join columns.
        drug_col: Column name containing drug IDs.
        geno_col: Column name containing the ``GENE_<variant>`` join key.
            (Note: legacy callers may provide a different name; the dataset
            uses ``geno_key`` if present, otherwise ``geno_col``.)
        target_cols: Target column names.
        multilabel_cols: Subset of ``target_cols`` that are multi-label.
        encoders: Pre-fitted encoders (REQUIRED for val/test sets to keep
            the same class layout as training).
        drug_lib / variant_lib: Override library paths (defaults from config).
        preload_ram: If True, eagerly loads all unique graphs into RAM.
        input_dimensions: Legacy nested-dict dim spec; converts to ``GraphDims``.
        type_data: Unused, kept for back-compat.
        inference_mode: If True, preserves identifying metadata on returned graphs.
    """

    def __init__(
        self,
        df: pl.DataFrame,
        drug_col: str,
        geno_col: str,
        target_cols: list[str],
        multilabel_cols: Iterable[str] | Set[str],
        encoders: dict[str, Any] | None = None,
        drug_lib: Path = LIBRARY / "drugs",
        variant_lib: Path = LIBRARY / "gene_graphs",
        preload_ram: bool = False,
        input_dimensions: dict[str, dict[str, int]] | None = None,
        type_data: str | None = None,  # noqa: ARG002 (legacy arg)
        inference_mode: bool = False,
    ) -> None:
        # 1. Frame
        if isinstance(df, pl.LazyFrame):
            logger.info("Collecting LazyFrame for Dataset access ...")
            self.df = df.collect()
        elif isinstance(df, pl.DataFrame):
            self.df = df
        else:
            msg = f"df must be a Polars DataFrame, got {type(df).__name__}"
            raise TypeError(msg)

        if input_dimensions:
            _validate_input_dims(input_dimensions)
        self.dims = _dims_from_input(input_dimensions)

        if preload_ram and len(self.df) > PRELOAD_THRESHOLD:
            logger.warning(
                "preload_ram=True with %d samples may cause OOM "
                "(threshold: %d).", len(self.df), PRELOAD_THRESHOLD,
            )

        self.drug_col = drug_col
        self.geno_col = geno_col
        self.target_cols = target_cols
        self.multilabel_cols = set(multilabel_cols) if multilabel_cols else set()
        self.inference_mode = inference_mode

        # 2. Indices + cache
        drug_index = GraphIndexBuilder.build_drug_index(drug_lib)
        variant_index = GraphIndexBuilder.build_gene_variant_index(variant_lib)
        logger.info(
            "Indexed %d drugs, %d variants",
            len(drug_index), sum(len(v) for v in variant_index.values()),
        )
        self.cache = GraphCache(
            drug_index=drug_index,
            variant_index=variant_index,
            dims=self.dims,
            inference_mode=inference_mode,
        )

        # 3. Encoders + targets
        self.target_encoder = TargetEncoder(
            target_cols=target_cols,
            multilabel_cols=self.multilabel_cols,
            encoders=dict(encoders) if encoders else None,
        )
        self.targets = self.target_encoder.fit_transform(self.df)

        # 4. Random-access lookups (avoid Polars indexing per __getitem__)
        self.lookup_drugs = self.df[self.drug_col].to_list()
        # Prefer the canonical 'geno_key' built by PharmacogenomicCleaner; fall
        # back to whatever the caller passed as geno_col.
        join_col = "geno_key" if "geno_key" in self.df.columns else self.geno_col
        self.lookup_genos = self.df[join_col].to_list()

        # 5. Optional preload
        if preload_ram:
            unique_drugs = self.df.select(
                pl.col(self.drug_col).unique().cast(pl.String)
            ).to_series().to_list()
            unique_genos = self.df.select(
                pl.col(join_col).unique().cast(pl.String)
            ).to_series().to_list()
            logger.info(
                "Preloading %d drugs and %d variants into RAM ...",
                len(unique_drugs), len(unique_genos),
            )
            self.cache.preload_drugs(unique_drugs)
            self.cache.preload_variants(unique_genos)
            logger.info(
                "Cached %d drugs, %d variants",
                self.cache.cached_drug_count, self.cache.cached_variant_count,
            )

    # ----- back-compat: callers expect .encoders --------------------------- #

    @property
    def encoders(self) -> dict[str, Any]:
        return self.target_encoder.encoders

    # ----- Dataset API ----------------------------------------------------- #

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        drug = self.cache.get_drug(str(self.lookup_drugs[idx]))
        geno = self.cache.get_variant(str(self.lookup_genos[idx]))
        targets = {col: self.targets[col][idx] for col in self.target_cols}
        return {"drug_data": drug, "geno_data": geno, "targets": targets}

    def get_cache_stats(self) -> dict[str, int | float]:
        """Cache hit/miss counters and rates (for logging / dashboards)."""
        return self.cache.stats()


# Convenience re-export so legacy ``from src.data.datasets import UNKNOWN_CATEGORY_LABEL``
# keeps working.
__all__ = [
    "DEFAULT_DIMENSIONS",
    "DoubleTowerDataset",
    "PGenDataset",
    "PGenProcessor",
    "PRELOAD_THRESHOLD",
    "UNKNOWN_CATEGORY_LABEL",
]
