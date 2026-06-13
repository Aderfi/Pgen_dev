"""Tests for src.data.library.haplotype_function — per-allele PGx function."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from src.data.library.haplotype_function import (
    FUNC_STATUSES,
    PATH_FUNCTION_DIM,
    HaplotypeFunctionProvider,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def star_tsv(tmp_path: Path) -> Path:
    path = tmp_path / "star_alleles.tsv"
    pl.DataFrame(
        {
            "gene": ["CYP2D6", "CYP2D6", "CYP2D6"],
            "allele": ["1", "4", "9"],
            "rsids": ["", "rs3892097", "rs5030655"],
            "function": ["normal_function", "no_function", "decreased_function"],
            "notes": ["AS 1.0", "AS 0.0", "AS 0.5"],
        }
    ).write_csv(path, separator="\t")
    return path


def test_dim() -> None:
    assert PATH_FUNCTION_DIM == 6 == len(FUNC_STATUSES) + 2


def test_no_function_allele(star_tsv: Path) -> None:
    provider = HaplotypeFunctionProvider.from_tsv(star_tsv)
    vec = provider.vector_for("CYP2D6", "*4")
    assert vec[FUNC_STATUSES.index("no_function")] == 1.0
    assert vec[4] == 0.0  # activity score (AS 0.0)
    assert vec[5] == 1.0  # pgx_known
    assert provider.misses == 0


def test_normal_function_activity(star_tsv: Path) -> None:
    provider = HaplotypeFunctionProvider.from_tsv(star_tsv)
    vec = provider.vector_for("CYP2D6", "*1")
    assert vec[FUNC_STATUSES.index("normal_function")] == 1.0
    assert vec[4] == 1.0  # AS 1.0


def test_sub_allele_inherits_core(star_tsv: Path) -> None:
    provider = HaplotypeFunctionProvider.from_tsv(star_tsv)
    assert provider.vector_for("CYP2D6", "*4.001") == provider.vector_for(
        "CYP2D6", "*4"
    )


def test_miss_returns_zeros(star_tsv: Path) -> None:
    provider = HaplotypeFunctionProvider.from_tsv(star_tsv)
    vec = provider.vector_for("CYP2D6", "*99")
    assert vec == [0.0] * PATH_FUNCTION_DIM
    assert provider.misses == 1


def test_null_provider() -> None:
    provider = HaplotypeFunctionProvider.null()
    assert provider.vector_for("CYP2D6", "*4") == [0.0] * PATH_FUNCTION_DIM
