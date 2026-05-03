"""Tests for src.data.library.chromosome."""

import pytest

from src.data.library.chromosome import CHROM_TO_REFSEQ, matches_fasta, to_refseq


class TestToRefseq:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("chr1", "NC_000001.11"),
            ("1", "NC_000001.11"),
            ("17", "NC_000017.11"),
            ("X", "NC_000023.11"),
            ("chrM", "NC_012920.1"),
        ],
    )
    def test_known_chromosomes(self, raw: str, expected: str) -> None:
        assert to_refseq(raw) == expected

    def test_unknown_passes_through(self) -> None:
        # Alt contigs / scaffolds aren't in the map; expect input back.
        assert to_refseq("KI270728.1") == "KI270728.1"

    def test_all_autosomes_covered(self) -> None:
        for n in range(1, 23):
            assert str(n) in CHROM_TO_REFSEQ
        for sex in ("X", "Y", "MT"):
            assert sex in CHROM_TO_REFSEQ


class TestMatchesFasta:
    def test_prefers_existing_canonical(self) -> None:
        keys = {"1", "2", "X"}
        assert matches_fasta("chr1", keys) == "1"

    def test_falls_back_to_refseq(self) -> None:
        keys = {"NC_000001.11", "NC_000023.11"}
        assert matches_fasta("chr1", keys) == "NC_000001.11"

    def test_returns_none_when_absent(self) -> None:
        assert matches_fasta("chr99", {"1", "2"}) is None
