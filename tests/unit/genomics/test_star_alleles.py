"""Tests for src.genomics.star_alleles."""

from pathlib import Path

import pytest

from src.domain.gene import AlleleFunction
from src.genomics.star_alleles import (
    StarAlleleMap,
    StarAlleleRecord,
    get_default_map,
    load_star_alleles,
)


@pytest.fixture
def tmp_tsv(tmp_path: Path) -> Path:
    p = tmp_path / "alleles.tsv"
    p.write_text(
        "gene\tallele\trsids\tfunction\tnotes\n"
        "CYP2D6\t4\trs3892097\tno_function\tAS 0.0\n"
        "CYP2D6\t10\trs1065852\tdecreased_function\tAS 0.25\n"
        "SLCO1B1\t15\trs4149056|rs2306283\tno_function\thaplotype\n"
        "NEW1\t1\t\tunknown\t\n"  # no rsids, blank notes
        "\t\t\t\t\n"  # blank row should be skipped
    )
    return p


class TestLoad:
    def test_loads_records(self, tmp_tsv: Path) -> None:
        m = load_star_alleles(tmp_tsv)
        assert len(m) == 4

    def test_skips_blank_rows(self, tmp_tsv: Path) -> None:
        m = load_star_alleles(tmp_tsv)
        assert "*" not in [
            label.split("*")[0] for label in m.labels
        ]  # no '*4' from blank row

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_star_alleles(tmp_path / "nope.tsv")

    def test_unknown_function_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.tsv"
        bad.write_text(
            "gene\tallele\trsids\tfunction\tnotes\n"
            "CYP2D6\t4\trs3892097\tweird_function\t\n"
        )
        with pytest.raises(ValueError, match="unknown function"):
            load_star_alleles(bad)


class TestStarAlleleMap:
    def test_lookup_by_label(self, tmp_tsv: Path) -> None:
        m = load_star_alleles(tmp_tsv)
        rec = m["CYP2D6*4"]
        assert rec.star_allele.function is AlleleFunction.NO_FUNCTION
        assert rec.rsids == ("rs3892097",)

    def test_haplotype_rsids_split(self, tmp_tsv: Path) -> None:
        m = load_star_alleles(tmp_tsv)
        rec = m["SLCO1B1*15"]
        assert rec.rsids == ("rs4149056", "rs2306283")

    def test_alleles_for_rsid(self, tmp_tsv: Path) -> None:
        m = load_star_alleles(tmp_tsv)
        # rs4149056 is shared across multiple alleles (in real catalog) but
        # only SLCO1B1*15 in this fixture.
        result = m.alleles_for_rsid("rs4149056")
        assert len(result) == 1
        assert result[0].label == "SLCO1B1*15"

    def test_alleles_for_unknown_rsid_empty(self, tmp_tsv: Path) -> None:
        m = load_star_alleles(tmp_tsv)
        assert m.alleles_for_rsid("rs999999999") == []

    def test_alleles_for_gene_case_insensitive(self, tmp_tsv: Path) -> None:
        m = load_star_alleles(tmp_tsv)
        upper = m.alleles_for_gene("CYP2D6")
        lower = m.alleles_for_gene("cyp2d6")
        assert {a.label for a in upper} == {a.label for a in lower}
        assert {a.label for a in upper} == {"CYP2D6*4", "CYP2D6*10"}

    def test_contains(self, tmp_tsv: Path) -> None:
        m = load_star_alleles(tmp_tsv)
        assert "CYP2D6*4" in m
        assert "FAKE*99" not in m

    def test_rsid_to_labels_inverse(self, tmp_tsv: Path) -> None:
        m = load_star_alleles(tmp_tsv)
        inv = m.rsid_to_labels
        assert inv["rs3892097"] == ["CYP2D6*4"]
        assert "SLCO1B1*15" in inv["rs4149056"]
        assert "SLCO1B1*15" in inv["rs2306283"]

    def test_duplicate_label_rejected(self, tmp_path: Path) -> None:
        dup = tmp_path / "dup.tsv"
        dup.write_text(
            "gene\tallele\trsids\tfunction\tnotes\n"
            "CYP2D6\t4\trs3892097\tno_function\t\n"
            "CYP2D6\t4\trs3892097\tno_function\t\n"
        )
        with pytest.raises(ValueError, match="duplicate"):
            load_star_alleles(dup)


class TestDefaultCatalog:
    def test_default_is_cached_singleton(self) -> None:
        a = get_default_map()
        b = get_default_map()
        assert a is b
        assert isinstance(a, StarAlleleMap)
        assert len(a) > 0

    def test_default_contains_known_alleles(self) -> None:
        m = get_default_map()
        # Sanity check — these are core CPIC alleles, expected in any catalog.
        for label in ("CYP2D6*4", "CYP2C19*2", "DPYD*2A"):
            assert label in m, f"{label} missing from default catalog"


class TestStarAlleleRecord:
    def test_normalize_rsids_string(self) -> None:
        from src.domain.gene import Gene, StarAllele

        rec = StarAlleleRecord(
            star_allele=StarAllele(gene=Gene(symbol="CYP2D6"), allele="4"),
            rsids="rs1|rs2 | rs3 ",
        )
        assert rec.rsids == ("rs1", "rs2", "rs3")

    def test_normalize_rsids_iterable(self) -> None:
        from src.domain.gene import Gene, StarAllele

        rec = StarAlleleRecord(
            star_allele=StarAllele(gene=Gene(symbol="CYP2D6"), allele="4"),
            rsids=["rs1", "rs2"],
        )
        assert rec.rsids == ("rs1", "rs2")
