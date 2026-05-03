"""Pharmacogenomic-specific data cleaning.

The pre-refactor ``DataLoaderUtils.clean_and_prepare_data`` did three things
that don't really belong together:

1. Drop rows with missing ``gene`` / ``genotype``.
2. Build the synthetic ``geno_key`` column by combining gene + star-allele
   evidence + rsID lookups (the project's join key against the gene-graph
   library).
3. Apply multi-label normalization and add a stratification column.

This module isolates step 2 — the only piece of the cleaning pipeline that's
biology-aware — into ``GenoKeyBuilder``, then composes everything in
``PharmacogenomicCleaner.clean()``. ``GenoKeyBuilder`` reads the rsID →
star-allele table from ``src.genomics.star_alleles`` so the lookup tracks the
canonical catalog rather than a hardcoded copy.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import polars as pl

from src.data.normalize import MultiLabelNormalizer, Stratifier
from src.genomics.star_alleles import get_default_map

logger = logging.getLogger(__name__)


_GENOTYPE_PREFIX_REGEX = r"^REF_SEQ\|"
_GENO_KEY_OUTPUT = "geno_key"


def _generate_keys_for_row(
    gene: str,
    genotype: str,
    alleles: str,
    rsid_to_labels: dict[str, list[str]],
) -> list[str]:
    """Compute the candidate ``GENE_<variant>`` keys for a single row.

    Priority:
        1. Star alleles literally present in the ``alleles`` column.
        2. rsIDs in the ``genotype`` column resolved through ``rsid_to_labels``.
        3. Fallback to ``GENE_<first rsID>`` if nothing else matched.
    """
    keys: set[str] = set()

    if alleles and "*" in alleles:
        for part in alleles.split("/"):
            stripped = part.strip()
            if "*" in stripped:
                keys.add(f"{gene}_{stripped}")

    parts = [p.strip() for p in genotype.split("|") if p.strip()]
    for rsid in parts:
        if rsid in rsid_to_labels:
            for star in rsid_to_labels[rsid]:
                if "*" in star:
                    suffix = "*" + star.split("*")[-1]
                    keys.add(f"{gene}_{suffix}")
                else:
                    keys.add(f"{gene}_{star}")
        elif not keys:
            keys.add(f"{gene}_{rsid}")

    if not keys and parts:
        keys.add(f"{gene}_{parts[0]}")

    return list(keys)


class GenoKeyBuilder:
    """Compute the project's join-key column ``geno_key`` for one or more rows.

    Captures the rsID lookup table at construction so each row uses a
    consistent snapshot. Defaults to the catalog at
    ``data/dicts/star_alleles.tsv``.
    """

    def __init__(self, rsid_to_labels: dict[str, list[str]] | None = None) -> None:
        self.rsid_to_labels = (
            rsid_to_labels
            if rsid_to_labels is not None
            else get_default_map().rsid_to_labels
        )

    def keys_for(self, gene: str, genotype: str, alleles: str = "") -> list[str]:
        return _generate_keys_for_row(
            gene=gene, genotype=genotype, alleles=alleles,
            rsid_to_labels=self.rsid_to_labels,
        )

    def add_keys_column(self, df: pl.DataFrame) -> pl.DataFrame:
        """Append the ``geno_key`` column to ``df`` and explode list values."""

        def _generate(row: dict) -> list[str]:
            return self.keys_for(
                gene=row["gene"], genotype=row["genotype"], alleles=row["alleles"]
            )

        with_keys = df.with_columns(
            pl.struct(["gene", "genotype", "alleles"])
            .map_elements(_generate, return_dtype=pl.List(pl.String))
            .alias(_GENO_KEY_OUTPUT)
        )
        return with_keys.explode(_GENO_KEY_OUTPUT).unique()


class PharmacogenomicCleaner:
    """Run the full cleaning pipeline for the training DataFrame.

    Composition over inheritance: this class wires together the small
    single-purpose utilities above. Callers can substitute a custom
    ``GenoKeyBuilder`` (e.g. for tests or to use a non-default star-allele map).
    """

    def __init__(
        self,
        *,
        multi_label_cols: Iterable[str] = (),
        key_builder: GenoKeyBuilder | None = None,
    ) -> None:
        self.multi_label_cols = list(multi_label_cols)
        self.key_builder = key_builder or GenoKeyBuilder()

    def clean(
        self,
        df: pl.DataFrame,
        *,
        stratify_col: str | list[str] | None = None,
    ) -> pl.DataFrame:
        """Drop blanks, build geno_key, normalize multi-labels, optionally stratify."""
        count_in = len(df)

        clean_gene = pl.col("gene").cast(pl.String).str.strip_chars()
        clean_genotype = (
            pl.col("genotype").cast(pl.String).str.strip_chars()
            .str.replace(_GENOTYPE_PREFIX_REGEX, "")
        )

        alleles_expr = (
            pl.col("alleles").fill_null("").str.strip_chars()
            if "alleles" in df.columns
            else pl.lit("").alias("alleles")
        )

        work = df.filter(
            pl.col("gene").is_not_null()
            & pl.col("genotype").is_not_null()
            & (clean_gene != "")
            & (clean_genotype != "")
        ).with_columns(
            [clean_gene.alias("gene"), clean_genotype.alias("genotype"), alleles_expr]
        )
        logger.info("Cleaner: dropped %d invalid rows.", count_in - len(work))

        work = self.key_builder.add_keys_column(work)
        logger.debug("Cleaner: %d rows after geno_key expansion.", len(work))

        ml_exprs = [
            MultiLabelNormalizer.normalize_expr(c).alias(c)
            for c in self.multi_label_cols
            if c in work.columns
        ]
        if ml_exprs:
            work = work.with_columns(ml_exprs)

        if stratify_col:
            cols = [stratify_col] if isinstance(stratify_col, str) else list(stratify_col)
            work = Stratifier.add_stratify_column(work, cols)

        logger.info("Cleaner: produced %d rows with geno_key.", len(work))
        return work
