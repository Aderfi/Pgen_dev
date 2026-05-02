"""Tests for src.domain.variant."""

import pytest

from src.domain.variant import (
    GenomeBuild,
    Genotype,
    Position,
    Variant,
    VariantType,
    Zygosity,
    normalize_chromosome,
)


class TestNormalizeChromosome:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("chr1", "1"),
            ("CHR22", "22"),
            ("1", "1"),
            ("X", "X"),
            ("chrX", "X"),
            ("MT", "MT"),
            ("M", "MT"),
            ("mtDNA", "MT"),
            ("NC_000001.11", "1"),
            ("NC_000023.11", "X"),
            ("NC_000012920.1", "MT"),
            ("KI270728.1", "KI270728.1"),  # alt contig passes through
        ],
    )
    def test_normalize_known_forms(self, raw: str, expected: str) -> None:
        assert normalize_chromosome(raw) == expected

    @pytest.mark.parametrize("bad", ["", "   ", "\n"])
    def test_rejects_empty(self, bad: str) -> None:
        with pytest.raises(ValueError, match="empty"):
            normalize_chromosome(bad)


class TestPosition:
    def test_construct_default_build(self) -> None:
        p = Position(chrom="chr1", pos=12345)
        assert p.chrom == "1"
        assert p.pos == 12345
        assert p.build is GenomeBuild.GRCH38

    def test_zero_position_rejected(self) -> None:
        with pytest.raises(ValueError):
            Position(chrom="1", pos=0)

    def test_negative_position_rejected(self) -> None:
        with pytest.raises(ValueError):
            Position(chrom="1", pos=-1)

    def test_str_representation_includes_build(self) -> None:
        p = Position(chrom="X", pos=100, build=GenomeBuild.GRCH37)
        assert str(p) == "GRCh37:X:100"

    def test_frozen(self) -> None:
        p = Position(chrom="1", pos=100)
        with pytest.raises(Exception):  # ValidationError or AttributeError depending on Pydantic mode
            p.pos = 200  # type: ignore[misc]


class TestVariant:
    def _pos(self) -> Position:
        return Position(chrom="17", pos=43044295)

    def test_snp_inferred(self) -> None:
        v = Variant(position=self._pos(), ref="G", alt="A")
        assert v.variant_type is VariantType.SNP

    def test_insertion_inferred(self) -> None:
        v = Variant(position=self._pos(), ref="A", alt="ATC")
        assert v.variant_type is VariantType.INSERTION

    def test_deletion_inferred(self) -> None:
        v = Variant(position=self._pos(), ref="ATC", alt="A")
        assert v.variant_type is VariantType.DELETION

    def test_mnp_inferred(self) -> None:
        v = Variant(position=self._pos(), ref="AC", alt="GT")
        assert v.variant_type is VariantType.MNP

    def test_lowercase_alleles_normalized(self) -> None:
        v = Variant(position=self._pos(), ref="g", alt="a")
        assert v.ref == "G"
        assert v.alt == "A"

    def test_invalid_allele_rejected(self) -> None:
        with pytest.raises(ValueError, match="invalid allele"):
            Variant(position=self._pos(), ref="GZX", alt="A")

    def test_explicit_type_preserved(self) -> None:
        v = Variant(
            position=self._pos(), ref="G", alt="A", variant_type=VariantType.STAR_ALLELE
        )
        assert v.variant_type is VariantType.STAR_ALLELE

    def test_rsid_validation(self) -> None:
        v = Variant(position=self._pos(), ref="G", alt="A", rsid="rs1801133")
        assert v.rsid == "rs1801133"

    def test_invalid_rsid_rejected(self) -> None:
        with pytest.raises(ValueError, match="invalid rsID"):
            Variant(position=self._pos(), ref="G", alt="A", rsid="1801133")


class TestGenotype:
    def test_construct(self) -> None:
        v = Variant(position=Position(chrom="1", pos=100), ref="A", alt="G")
        gt = Genotype(variant=v, zygosity=Zygosity.HETEROZYGOUS, sample_id="patient_42")
        assert gt.zygosity is Zygosity.HETEROZYGOUS
        assert gt.sample_id == "patient_42"
