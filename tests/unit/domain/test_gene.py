"""Tests for src.domain.gene."""

import pytest

from src.domain.schemas.gene import AlleleFunction, Gene, StarAllele


class TestGene:
    def test_uppercase_normalization(self) -> None:
        g = Gene(symbol="cyp2d6")
        assert g.symbol == "CYP2D6"

    def test_strip_whitespace(self) -> None:
        g = Gene(symbol="  CYP2D6  ")
        assert g.symbol == "CYP2D6"

    def test_invalid_symbol_rejected(self) -> None:
        with pytest.raises(ValueError, match="invalid HGNC symbol"):
            Gene(symbol="123ABC")

    def test_ensembl_optional(self) -> None:
        g = Gene(symbol="CYP2D6", ensembl_id="ENSG00000100197")
        assert g.ensembl_id == "ENSG00000100197"

    def test_ensembl_with_version(self) -> None:
        g = Gene(symbol="CYP2D6", ensembl_id="ENSG00000100197.5")
        assert g.ensembl_id == "ENSG00000100197.5"

    def test_invalid_ensembl_rejected(self) -> None:
        with pytest.raises(ValueError, match="invalid Ensembl"):
            Gene(symbol="CYP2D6", ensembl_id="ENS123")


class TestStarAllele:
    def test_parse_basic(self) -> None:
        sa = StarAllele.parse("CYP2D6*4")
        assert sa.gene.symbol == "CYP2D6"
        assert sa.allele == "4"
        assert sa.function is AlleleFunction.UNKNOWN

    def test_parse_with_function(self) -> None:
        sa = StarAllele.parse("CYP2D6*4", function=AlleleFunction.NO_FUNCTION)
        assert sa.function is AlleleFunction.NO_FUNCTION

    def test_parse_subvariant(self) -> None:
        sa = StarAllele.parse("CYP2D6*4.1")
        assert sa.allele == "4.1"

    def test_parse_letter_suffix(self) -> None:
        sa = StarAllele.parse("CYP2D6*4A")
        assert sa.allele == "4A"

    def test_parse_complex(self) -> None:
        sa = StarAllele.parse("HLA-B*57.01")
        assert sa.gene.symbol == "HLA-B"
        assert sa.allele == "57.01"

    @pytest.mark.parametrize("bad", ["CYP2D6", "*4", "CYP2D6*", "no asterisk", ""])
    def test_parse_rejects_bad_labels(self, bad: str) -> None:
        with pytest.raises(ValueError):
            StarAllele.parse(bad)

    def test_label_round_trip(self) -> None:
        sa = StarAllele.parse("CYP2D6*4")
        assert sa.label == "CYP2D6*4"
        assert str(sa) == "CYP2D6*4"

    def test_strip_leading_asterisk_in_allele(self) -> None:
        sa = StarAllele(gene=Gene(symbol="CYP2D6"), allele="*17")
        assert sa.allele == "17"

    def test_invalid_allele_format(self) -> None:
        with pytest.raises(ValueError, match="invalid star allele"):
            StarAllele(gene=Gene(symbol="CYP2D6"), allele="XYZ")
