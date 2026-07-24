"""Tabular data loaders.

A focused replacement for the loading half of ``DataLoaderUtils``. Reads
training/inference CSV/TSV files into Polars DataFrames with schema
overrides, column projection, and a unified set of null tokens.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


# Standardized null tokens we always treat as missing across CSV/TSV inputs.
DEFAULT_NULL_VALUES: list[str] = ["", "NA", "NaN", "null", "N/A"]


# Schema for the project's primary training file. Centralized so both the
# pipeline and the API agree on dtypes; any column not listed here is inferred.
TRAIN_DATA_SCHEMA: dict[str, Any] = {
    # Identifiers
    "drugs_cid": pl.String,
    "drugs": pl.String,
    "gene": pl.String,
    "genotype": pl.String,
    "alleles": pl.String,
    "phenotype_category": pl.String,
    # Pharmacogenetic annotation columns from PharmGKB var_drug_ann
    "metabolizer_types": pl.String,
    "population_types": pl.Categorical,
    "population_phenotypes_or_diseases": pl.String,
    "comparison_allele(s)_or_genotype(s)": pl.String,
    "comparison_metabolizer_types": pl.String,
    "significance": pl.Categorical,
    "is/is_not_associated": pl.Categorical,
    "variant_annotation_id": pl.Int64,
    "pmid": pl.Int64,
    "sentence": pl.String,
    "notes": pl.String,
}


class TabularLoader:
    """Read CSV/TSV inputs with the project-wide null-value contract.

    Stateless utility — instances are not required, ``TabularLoader.load(...)``
    can be called as a free function. Kept as a class for symmetry with
    ``Stratifier`` and ``MultiLabelNormalizer`` and to make it easy to
    parametrize per-call (e.g. test fixtures with different separators).
    """

    # Sentinel that means "use TRAIN_DATA_SCHEMA". We can't use TRAIN_DATA_SCHEMA
    # itself as the default because callers want a way to opt out (None) and a
    # way to pass a custom schema (a dict).
    _DEFAULT_SCHEMA = object()

    @staticmethod
    def load(
        path: str | Path,
        *,
        columns: list[str] | None = None,
        schema: dict[str, Any] | None | object = _DEFAULT_SCHEMA,
        null_values: list[str] | None = None,
        separator: str | None = None,
    ) -> pl.DataFrame:
        """Read ``path`` (CSV or TSV) into a Polars DataFrame.

        - ``separator`` defaults to ``\\t`` for non-``.csv`` files.
        - ``schema`` left unspecified means TRAIN_DATA_SCHEMA. Pass ``None``
          to disable the schema entirely (let Polars infer all dtypes), or
          pass a custom dict to override.
        - ``null_values`` defaults to ``DEFAULT_NULL_VALUES``.

        Raises whatever Polars raises on invalid input — callers are
        expected to wrap with their domain-specific error type.
        """
        path_str = str(path)
        sep = (
            separator
            if separator is not None
            else ("," if path_str.endswith(".csv") else "\t")
        )
        if schema is TabularLoader._DEFAULT_SCHEMA:
            schema_overrides = TRAIN_DATA_SCHEMA
        else:
            schema_overrides = schema  # may be None or a custom dict

        # Polars' `schema` arg requires ALL columns to be listed; use
        # `schema_overrides` (partial dtype overrides) for our use case.
        return pl.read_csv(
            path_str,
            separator=sep,
            has_header=True,
            columns=columns,
            schema_overrides=schema_overrides,
            null_values=null_values if null_values is not None else DEFAULT_NULL_VALUES,
            encoding="utf-8",
        )
