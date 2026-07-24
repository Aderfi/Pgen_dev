"""Tests for src.data.library.ingest.hgvs_build — VCF coords → genomic HGVS."""

from __future__ import annotations

import pytest

from src.core.exceptions import BioinformaticsError
from src.data.library.ingest.hgvs_build import genomic_hgvs, genomic_hgvs_body
from src.domain.schemas.hgvs import MolecularType, VariantKind
from src.genomics.hgvs_parser import parse


class TestBody:
    @pytest.mark.parametrize(
        ("pos", "ref", "alt", "expected"),
        [
            (42128945, "C", "T", "g.42128945C>T"),  # SNV
            (100, "AC", "A", "g.101del"),  # single-base deletion (trim prefix)
            (100, "ACGT", "A", "g.101_103del"),  # multi-base deletion
            (100, "A", "AC", "g.100_101insC"),  # insertion
            (100, "AC", "GT", "g.100_101delinsGT"),  # delins
            (100, "AT", "AG", "g.101T>G"),  # MNV collapses to SNV
            (100, "A", "ACGT", "g.100_101insCGT"),  # multi-base insertion
        ],
    )
    def test_minimal_representation(
        self, pos: int, ref: str, alt: str, expected: str
    ) -> None:
        assert genomic_hgvs_body(pos, ref, alt) == expected

    def test_ref_equals_alt_raises(self) -> None:
        with pytest.raises(BioinformaticsError, match="not a variant"):
            genomic_hgvs_body(100, "A", "A")

    def test_empty_allele_raises(self) -> None:
        with pytest.raises(BioinformaticsError, match="empty"):
            genomic_hgvs_body(100, "", "A")


class TestFullExpression:
    @pytest.mark.parametrize(
        ("chrom", "accession"),
        [("22", "NC_000022.11"), ("chr22", "NC_000022.11"), ("X", "NC_000023.11")],
    )
    def test_chrom_resolves_to_refseq(self, chrom: str, accession: str) -> None:
        assert genomic_hgvs(chrom, 100, "C", "T") == f"{accession}:g.100C>T"


class TestParserRoundTrip:
    """Every produced expression must be valid HGVS the project parser accepts."""

    @pytest.mark.parametrize(
        ("pos", "ref", "alt", "kind"),
        [
            (42128945, "C", "T", VariantKind.SUBSTITUTION),
            (100, "AC", "A", VariantKind.DELETION),
            (100, "ACGT", "A", VariantKind.DELETION),
            (100, "A", "AC", VariantKind.INSERTION),
            (100, "AC", "GT", VariantKind.DELINS),
        ],
    )
    def test_roundtrip(self, pos: int, ref: str, alt: str, kind: VariantKind) -> None:
        variant = parse(genomic_hgvs("22", pos, ref, alt))
        assert variant.molecular_type is MolecularType.GENOMIC
        assert variant.reference_sequence == "NC_000022.11"
        assert variant.primary_change.kind is kind

    def test_substitution_alleles_preserved(self) -> None:
        change = parse(genomic_hgvs("22", 42128945, "C", "T")).primary_change
        assert change.reference_allele == "C"
        assert change.alternate_allele == "T"
        assert change.start.base == 42128945
