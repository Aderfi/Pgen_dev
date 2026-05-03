"""Console-facing IO helpers and a back-compat facade over the data pipeline.

This module is the historical home of ``DataLoaderUtils``. The actual
implementations now live under ``src.data.*``:

- ``src.data.loaders.TabularLoader``     reads CSV/TSV with the project schema.
- ``src.data.cleaning.PharmacogenomicCleaner``  builds ``geno_key`` + cleans.
- ``src.data.normalize.MultiLabelNormalizer`` / ``Stratifier``.
- ``src.data.graph_indexing.GraphIndexBuilder``  walks the on-disk library.

``DataLoaderUtils`` here remains as a thin compatibility shim used by
``src.pipeline``; new code should import from the focused modules above.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from src.config.manager import DIRS, METADATA, MULTI_LABEL_COLS, VERSION
from src.data.cleaning import PharmacogenomicCleaner
from src.data.graph_indexing import GraphIndexBuilder
from src.data.loaders import TRAIN_DATA_SCHEMA, TabularLoader
from src.data.normalize import MultiLabelNormalizer, Stratifier
from src.genomics.star_alleles import get_default_map as _get_star_map

LOGS_DIR = DIRS["logs"]
UNKNOWN_TOKEN = "__UNKNOWN__"

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# JSON helpers — small enough to stay here.
# --------------------------------------------------------------------------- #


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """Write ``data`` as JSON with 2-space indent."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON file into a dict."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Console banners and GPL notices.
# --------------------------------------------------------------------------- #


def welcome_message() -> None:
    msg = f"""
    ============================================
            PHARMAGEN v{VERSION}
    ============================================
    Pharmacogenetic prediction with deep learning.

    Logs: {LOGS_DIR}
    ============================================
    """
    print(msg)


def print_gnu_notice() -> None:
    """Print the short GPL boot notice."""
    start_year = 2025
    current_year = datetime.now().year
    year_str = f"{start_year}-{current_year}" if current_year > start_year else str(start_year)
    author = "Adrim Hamed Outmani (@Aderfi)"
    program = METADATA.get("project_name", "Pharmagen")

    notice = f"""
    {program} Copyright (C) {year_str} {author}
    This program comes with ABSOLUTELY NO WARRANTY; for details type `show w'.
    This is free software, and you are welcome to redistribute it
    under certain conditions; type `show c' for details.
    """
    print(notice)


def print_warranty_details() -> None:
    """Full warranty text for the ``show w`` command."""
    print("\n" + "=" * 60)
    print("NO WARRANTY")
    print("=" * 60)
    print("""
    BECAUSE THE PROGRAM IS LICENSED FREE OF CHARGE, THERE IS NO WARRANTY
    FOR THE PROGRAM, TO THE EXTENT PERMITTED BY APPLICABLE LAW. EXCEPT WHEN
    OTHERWISE STATED IN WRITING THE COPYRIGHT HOLDERS AND/OR OTHER PARTIES
    PROVIDE THE PROGRAM "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESSED
    OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
    MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE ENTIRE RISK AS
    TO THE QUALITY AND PERFORMANCE OF THE PROGRAM IS WITH YOU. SHOULD THE
    PROGRAM PROVE DEFECTIVE, YOU ASSUME THE COST OF ALL NECESSARY SERVICING,
    REPAIR OR CORRECTION.
    """)
    input("\nPress [Enter] to return...")


def print_conditions_details() -> None:
    """Full redistribution-conditions text for the ``show c`` command."""
    print("\n" + "=" * 60)
    print("REDISTRIBUTION CONDITIONS")
    print("=" * 60)
    print("""
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <https://www.gnu.org/licenses/>.
    """)
    input("\nPress [Enter] to return...")


# --------------------------------------------------------------------------- #
# Star-allele compat shims (consumed by legacy pipeline code).
# --------------------------------------------------------------------------- #


_STAR_MAP = _get_star_map()
STAR_ALLELE_MAP: dict[str, str] = {
    label: "|".join(_STAR_MAP[label].rsids) for label in _STAR_MAP.labels
}
RSID_TO_STAR_ALLELES: dict[str, list[str]] = _STAR_MAP.rsid_to_labels


# --------------------------------------------------------------------------- #
# DataLoaderUtils — back-compat facade. New code should import from
# src.data.loaders / src.data.cleaning / src.data.normalize directly.
# --------------------------------------------------------------------------- #


class DataLoaderUtils:
    """Deprecated facade — use the focused classes under ``src.data.*`` instead.

    Kept so that ``src.pipeline.train_pipeline`` and any older notebooks keep
    working. Each method now delegates to the new module.
    """

    @staticmethod
    def load_dataframe(
        csv_path: str | Path,
        cols: list[str],
        stratify_col: list[str] | str | None = None,
    ) -> pl.DataFrame:
        try:
            df = TabularLoader.load(csv_path, columns=cols)
        except Exception as e:
            logger.error("Error reading CSV %s: %s", csv_path, e)
            raise
        return DataLoaderUtils.clean_and_prepare_data(df, stratify_col=stratify_col)

    @staticmethod
    def normalize_multilabel_col(col_name: str, delimiter: str = "|") -> pl.Expr:
        return MultiLabelNormalizer.normalize_expr(col_name, delimiter)

    @staticmethod
    def add_stratify_column(
        df: pl.DataFrame, stratify_cols: list[str]
    ) -> pl.DataFrame:
        return Stratifier.add_stratify_column(df, stratify_cols)

    @staticmethod
    def clean_and_prepare_data(
        df: pl.DataFrame, stratify_col: list[str] | str | None = None
    ) -> pl.DataFrame:
        cleaner = PharmacogenomicCleaner(multi_label_cols=MULTI_LABEL_COLS)
        return cleaner.clean(df, stratify_col=stratify_col)

    # --- legacy index builders — superseded by GraphIndexBuilder ---

    @staticmethod
    def _build_drug_index(drug_lib: Path) -> dict[str, Path]:
        return GraphIndexBuilder.build_drug_index(drug_lib)

    @staticmethod
    def _build_genes_index(variant_lib: Path) -> dict[str, dict[str, Path]]:
        return GraphIndexBuilder.build_gene_variant_index(variant_lib)


__all__ = [
    "DataLoaderUtils",
    "RSID_TO_STAR_ALLELES",
    "STAR_ALLELE_MAP",
    "TRAIN_DATA_SCHEMA",
    "UNKNOWN_TOKEN",
    "load_json",
    "print_conditions_details",
    "print_gnu_notice",
    "print_warranty_details",
    "save_json",
    "welcome_message",
]
