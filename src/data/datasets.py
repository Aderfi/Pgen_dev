"""PyTorch Datasets for Pharmagen.

:class:`DoubleTowerDataset` — the active two-tower GNN dataset. Composes
``GraphCache`` (in-RAM graph store) and ``TargetEncoder`` (target
tensorization).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Set
from pathlib import Path
from typing import Any

import polars as pl
from torch.utils.data import Dataset

from src.config import get_settings
from src.core import DataError
from src.data.cache import GraphCache, GraphDims
from src.data.encoders import TargetEncoder
from src.data.graph_indexing import GraphIndexBuilder

logger = logging.getLogger(__name__)

_LIBRARY = get_settings().paths.library

# Threshold above which RAM preloading is suspicious (warn, don't refuse).
PRELOAD_THRESHOLD = 5000

# Default tower dimensions when callers don't override.
DEFAULT_DIMENSIONS: dict[str, dict[str, int]] = {
    "drugs": {"features": 61, "edges": 18, "attrs": 0, "global": 1038, "admet": 41},
    "geno": {"features": 9, "edges": 3, "attrs": 0},
}


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
        drug_global=drugs.get("global", DEFAULT_DIMENSIONS["drugs"]["global"]),
        drug_admet=drugs.get("admet", DEFAULT_DIMENSIONS["drugs"]["admet"]),
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
                msg = (
                    f"invalid dimension {kind}.{subkey}: {v} (must be non-negative int)"
                )
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
        drug_lib: Path = _LIBRARY / "drugs",
        variant_lib: Path = _LIBRARY / "gene_graphs",
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
                "preload_ram=True with %d samples may cause OOM (threshold: %d).",
                len(self.df),
                PRELOAD_THRESHOLD,
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
            len(drug_index),
            sum(len(v) for v in variant_index.values()),
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
            unique_drugs = (
                self.df.select(pl.col(self.drug_col).unique().cast(pl.String))
                .to_series()
                .to_list()
            )
            unique_genos = (
                self.df.select(pl.col(join_col).unique().cast(pl.String))
                .to_series()
                .to_list()
            )
            logger.info(
                "Preloading %d drugs and %d variants into RAM ...",
                len(unique_drugs),
                len(unique_genos),
            )
            self.cache.preload_drugs(unique_drugs)
            self.cache.preload_variants(unique_genos)
            logger.info(
                "Cached %d drugs, %d variants",
                self.cache.cached_drug_count,
                self.cache.cached_variant_count,
            )

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


__all__ = [
    "DEFAULT_DIMENSIONS",
    "DoubleTowerDataset",
    "PRELOAD_THRESHOLD",
]
