"""Tests for src.data.library.geno_func — the assembled functional profile."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from src.data.library.consequence import CONSEQUENCE_DIM
from src.data.library.geno_func import (
    GENE_GLOBAL_DIM,
    GenoFuncProvider,
    load_hgvs_protein_table,
    load_star_allele_function,
    parse_activity_score,
)
from src.data.library.protein_change import PROTEIN_CHANGE_DIM

if TYPE_CHECKING:
    from pathlib import Path

# Block offsets within the assembled vector: A (function) | B (SO) | C (protein).
_FUNCTION_DIM = GENE_GLOBAL_DIM - CONSEQUENCE_DIM - PROTEIN_CHANGE_DIM  # 6
_FUNC_NO, _ACTIVITY, _PGX_KNOWN = 0, 4, 5
_B0 = _FUNCTION_DIM  # start of the SO block
_C0 = _FUNCTION_DIM + CONSEQUENCE_DIM  # start of the protein block


@pytest.fixture
def star_tsv(tmp_path: Path) -> Path:
    """A minimal star-allele table covering single- and multi-rsID rows."""
    path = tmp_path / "star_alleles.tsv"
    pl.DataFrame(
        {
            "gene": ["CYP2D6", "SLCO1B1", "SLCO1B1"],
            "allele": ["4", "15", "37"],
            "rsids": ["rs3892097", "rs4149056|rs2306283", "rs2306283"],
            "function": ["no_function", "no_function", "normal_function"],
            "notes": ["AS 0.0", "haplotype; AS 0.0", "AS 1.0"],
        }
    ).write_csv(path, separator="\t")
    return path


class TestGeneGlobalDim:
    def test_dimension_matches_models_toml(self) -> None:
        # Mirrored in src/config/data/models.toml [TwoTowerGAT].geno_global_features.
        assert GENE_GLOBAL_DIM == 27


class TestParseActivityScore:
    @pytest.mark.parametrize(
        "notes, function, expected",
        [
            ("AS 0.0", "no_function", 0.0),
            ("severe; AS 0.0", "decreased_function", 0.0),
            ("AS 1.0+", "increased_function", 1.0),
            (None, "decreased_function", 0.5),  # falls back to CPIC-style default
        ],
    )
    def test_parses_or_defaults(
        self, notes: str | None, function: str, expected: float
    ) -> None:
        assert parse_activity_score(notes, function) == expected


class TestLoadStarAlleleFunction:
    def test_skips_multi_rsid_haplotypes(self, star_tsv: Path) -> None:
        mapping = load_star_allele_function(star_tsv)
        assert mapping["rs3892097"] == ("no_function", 0.0)
        assert mapping["rs2306283"] == ("normal_function", 1.0)  # *37, not *15 combo
        assert "rs4149056" not in mapping


class TestLoadHgvsProteinTable:
    def test_reads_rsid_to_hgvs(self, tmp_path: Path) -> None:
        path = tmp_path / "hgvs.tsv"
        pl.DataFrame(
            {"rsid": ["rs1", "rs2"], "hgvs_p": ["p.Arg296Cys", "p.Asp36Tyr"]}
        ).write_csv(path, separator="\t")
        assert load_hgvs_protein_table(path) == {
            "rs1": "p.Arg296Cys",
            "rs2": "p.Asp36Tyr",
        }

    def test_missing_path_is_empty(self) -> None:
        assert load_hgvs_protein_table(None) == {}


class TestGenoFuncProvider:
    def test_function_layer_a(self, star_tsv: Path) -> None:
        prov = GenoFuncProvider.from_sources(star_tsv)
        vec = prov.vector_for("rs3892097", None)[0].tolist()
        assert vec[_FUNC_NO] == 1.0
        assert vec[_ACTIVITY] == 0.0
        assert vec[_PGX_KNOWN] == 1.0
        assert prov.activity_for("rs3892097") == 0.0

    def test_consequence_layer_b_from_fxn_class(self, star_tsv: Path) -> None:
        prov = GenoFuncProvider.from_sources(star_tsv)
        vec = prov.vector_for("rsUNKNOWN", "stop_gained,coding_sequence_variant")[0]
        block = vec[_B0 : _B0 + CONSEQUENCE_DIM].tolist()
        assert block[0] == 1.0  # stop_gained group
        assert block[-1] == 1.0  # consequence_known mask
        assert block[-2] == 1.0  # max-severity (stop_gained = 1.0)

    def test_protein_layer_c_from_hgvs_table(
        self, star_tsv: Path, tmp_path: Path
    ) -> None:
        hgvs = tmp_path / "hgvs.tsv"
        pl.DataFrame({"rsid": ["rs3892097"], "hgvs_p": ["p.Asp36Tyr"]}).write_csv(
            hgvs, separator="\t"
        )
        prov = GenoFuncProvider.from_sources(star_tsv, hgvs_table=hgvs)
        vec = prov.vector_for("rs3892097", None)[0]
        block = vec[_C0 : _C0 + PROTEIN_CHANGE_DIM].tolist()
        assert block[0] > 0.0  # grantham distance nonzero
        assert block[-1] == 1.0  # hgvs_protein_known mask

    def test_all_three_layers_compose(self, star_tsv: Path, tmp_path: Path) -> None:
        hgvs = tmp_path / "hgvs.tsv"
        pl.DataFrame({"rsid": ["rs3892097"], "hgvs_p": ["p.Arg296Ter"]}).write_csv(
            hgvs, separator="\t"
        )
        prov = GenoFuncProvider.from_sources(star_tsv, hgvs_table=hgvs)
        vec = prov.vector_for("rs3892097", "missense_variant")[0]
        assert vec.shape[0] == GENE_GLOBAL_DIM
        assert vec[_PGX_KNOWN] == 1.0  # A present
        assert vec[_B0 + CONSEQUENCE_DIM - 1] == 1.0  # B present (known mask)
        assert vec[_C0 + PROTEIN_CHANGE_DIM - 1] == 1.0  # C present (known mask)

    def test_no_signal_is_zero_and_tallied(self, star_tsv: Path) -> None:
        prov = GenoFuncProvider.from_sources(star_tsv)
        vec = prov.vector_for("rsUNKNOWN", None)
        assert vec.shape == (1, GENE_GLOBAL_DIM)
        assert float(vec.sum()) == 0.0
        assert prov.misses == 1

    def test_null_provider_is_fully_zero(self) -> None:
        # Disabled provider zeros everything, including the otherwise-free Layer B.
        prov = GenoFuncProvider.null()
        assert float(prov.vector_for("rs3892097", "stop_gained").sum()) == 0.0
        assert prov.activity_for("rs3892097") is None
