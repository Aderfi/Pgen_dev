"""Tests for src.data.library.geno_func — the genotype functional profile."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from src.data.library.geno_func import (
    GENE_GLOBAL_DIM,
    GenoFuncProvider,
    load_star_allele_function,
    parse_activity_score,
)

if TYPE_CHECKING:
    from pathlib import Path

# Layer offsets (kept in lockstep with the module's vector layout).
_FUNC_NO, _FUNC_DEC, _FUNC_NORMAL, _FUNC_INC = 0, 1, 2, 3
_ACTIVITY, _PGX_KNOWN = 4, 5
_AM, _AM_MASK, _CADD, _CADD_MASK = 6, 7, 8, 9


@pytest.fixture
def star_tsv(tmp_path: Path) -> Path:
    """A minimal star-allele table covering single- and multi-rsID rows."""
    path = tmp_path / "star_alleles.tsv"
    pl.DataFrame(
        {
            "gene": ["CYP2D6", "CYP2C19", "SLCO1B1", "SLCO1B1"],
            "allele": ["4", "17", "15", "37"],
            "rsids": ["rs3892097", "rs12248560", "rs4149056|rs2306283", "rs2306283"],
            "function": [
                "no_function",
                "increased_function",
                "no_function",
                "normal_function",
            ],
            "notes": ["AS 0.0", "AS 1.0+", "haplotype; AS 0.0", "AS 1.0"],
        }
    ).write_csv(path, separator="\t")
    return path


class TestGeneGlobalDim:
    def test_dimension_matches_models_toml(self) -> None:
        # Mirrored in src/config/data/models.toml [TwoTowerGAT].geno_global_features.
        assert GENE_GLOBAL_DIM == 10


class TestParseActivityScore:
    @pytest.mark.parametrize(
        "notes, function, expected",
        [
            ("AS 0.0", "no_function", 0.0),
            ("severe; AS 0.0", "decreased_function", 0.0),
            ("AS 1.0+", "increased_function", 1.0),
            (None, "decreased_function", 0.5),  # falls back to CPIC-style default
            ("no AS token here", "normal_function", 1.0),
        ],
    )
    def test_parses_or_defaults(
        self, notes: str | None, function: str, expected: float
    ) -> None:
        assert parse_activity_score(notes, function) == expected


class TestLoadStarAlleleFunction:
    def test_skips_multi_rsid_haplotypes(self, star_tsv: Path) -> None:
        mapping = load_star_allele_function(star_tsv)
        # Single-rsID rows are kept; the multi-rsID *15 haplotype is dropped, so
        # rs2306283 resolves to *37 (normal), never the *15 (no_function) combo.
        assert mapping["rs3892097"] == ("no_function", 0.0)
        assert mapping["rs2306283"] == ("normal_function", 1.0)
        assert "rs4149056" not in mapping


class TestGenoFuncProvider:
    def test_pgx_function_layer(self, star_tsv: Path) -> None:
        prov = GenoFuncProvider.from_sources(star_tsv)
        vec = prov.vector_for("rs3892097", "22", 42130692, "C", "T")[0].tolist()
        assert vec[_FUNC_NO] == 1.0
        assert vec[_ACTIVITY] == 0.0
        assert vec[_PGX_KNOWN] == 1.0
        assert prov.activity_for("rs3892097") == 0.0

    def test_pathogenicity_layer_by_coordinate(
        self, star_tsv: Path, tmp_path: Path
    ) -> None:
        am = tmp_path / "am.tsv"
        pl.DataFrame(
            {
                "chrom": ["chr22"],
                "pos": [42130692],
                "ref": ["C"],
                "alt": ["T"],
                "alphamissense": [0.97],
            }
        ).write_csv(am, separator="\t")
        cadd = tmp_path / "cadd.tsv"
        pl.DataFrame(
            {
                "chrom": ["22"],
                "pos": [42130692],
                "ref": ["C"],
                "alt": ["T"],
                "cadd_phred": [30.0],
            }
        ).write_csv(cadd, separator="\t")

        prov = GenoFuncProvider.from_sources(
            star_tsv, alphamissense_path=am, cadd_path=cadd
        )
        vec = prov.vector_for("rs3892097", "22", 42130692, "C", "T")[0].tolist()
        assert vec[_AM] == pytest.approx(0.97)
        assert vec[_AM_MASK] == 1.0
        assert vec[_CADD] == pytest.approx(30.0 / 40.0)  # PHRED capped/normalised
        assert vec[_CADD_MASK] == 1.0

    def test_unknown_variant_is_zero_and_tallied(self, star_tsv: Path) -> None:
        prov = GenoFuncProvider.from_sources(star_tsv)
        vec = prov.vector_for("rsUNKNOWN", None, None, None, None)
        assert vec.shape == (1, GENE_GLOBAL_DIM)
        assert float(vec.sum()) == 0.0
        assert prov.misses == 1

    def test_null_provider_yields_zero(self) -> None:
        prov = GenoFuncProvider.null()
        vec = prov.vector_for("rs3892097", "22", 42130692, "C", "T")
        assert float(vec.sum()) == 0.0
        assert prov.activity_for("rs3892097") is None
