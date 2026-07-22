"""Pharmacogenomic-specific data cleaning.

The pre-refactor ``DataLoaderUtils.clean_and_prepare_data`` bundled three jobs:

1. Drop rows with missing ``gene`` / ``genotype``.
2. Build a synthetic ``geno_key`` join column against the gene-graph library.
3. Apply multi-label normalization and add a stratification column.

Step 2 is gone. The genotype tower now keys on ``(gene, genotype)`` directly and
resolves each pair to a subgraph at access time via ``GenotypeResolver`` over the
single-file ``GenoLibrary`` — there is no per-variant join key to precompute. The
cleaner therefore just normalizes ``gene``/``genotype`` (and optional multi-label
targets) and leaves the biology to the resolver.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import polars as pl

from src.data.normalize import MultiLabelNormalizer, Stratifier

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)


_GENOTYPE_PREFIX_REGEX = r"^REF_SEQ\|"


class PharmacogenomicCleaner:
    """Normalize the training/inference DataFrame for the two-tower pipeline.

    Drops rows missing ``gene`` / ``genotype``, strips the legacy ``REF_SEQ|``
    genotype prefix, normalizes the configured multi-label target columns, and
    optionally adds a ``_stratify`` column. No join key is built — the genotype
    tower resolves ``(gene, genotype)`` on demand.
    """

    def __init__(self, *, multi_label_cols: Iterable[str] = ()) -> None:
        self.multi_label_cols = list(multi_label_cols)

    def clean(
        self,
        df: pl.DataFrame,
        *,
        stratify_col: str | list[str] | None = None,
    ) -> pl.DataFrame:
        """Drop blanks, normalize gene/genotype/multi-labels, optionally stratify."""
        count_in = len(df)

        clean_gene = pl.col("gene").cast(pl.String).str.strip_chars()
        clean_genotype = (
            pl.col("genotype")
            .cast(pl.String)
            .str.strip_chars()
            .str.replace(_GENOTYPE_PREFIX_REGEX, "")
        )

        work = df.filter(
            pl.col("gene").is_not_null()
            & pl.col("genotype").is_not_null()
            & (clean_gene != "")
            & (clean_genotype != "")
        ).with_columns([clean_gene.alias("gene"), clean_genotype.alias("genotype")])
        logger.info("Cleaner: dropped %d invalid rows.", count_in - len(work))

        ml_exprs = [
            MultiLabelNormalizer.normalize_expr(c).alias(c)
            for c in self.multi_label_cols
            if c in work.columns
        ]
        if ml_exprs:
            work = work.with_columns(ml_exprs)

        if stratify_col:
            cols = (
                [stratify_col] if isinstance(stratify_col, str) else list(stratify_col)
            )
            work = Stratifier.add_stratify_column(work, cols)

        logger.info("Cleaner: produced %d clean rows.", len(work))
        return work


__all__ = ["PharmacogenomicCleaner"]
