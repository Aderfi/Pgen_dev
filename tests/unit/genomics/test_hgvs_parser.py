"""Tests for src.genomics.hgvs_parser — repeated-sequence variants."""

from __future__ import annotations

import pytest

from src.core import BioinformaticsError
from src.domain.types import VariantKind
from src.genomics.hgvs_parser import parse

# Composite repeats found in the KG exports (PharmGKB alleles via NCBI).
REAL_COMPOSITE = [
    (
        "NC_000002.12:g.85554455_85554456T[5]GTT[1]",
        [("T", 5), ("GTT", 1)],
        8,
    ),
    (
        "NC_000007.14:g.97872311_97872327GGCGCGGGGCGCAG[4]"
        "GGCACGGGGCGCAGGGCGCGGGGCGCAGGGC[1]",
        [("GGCGCGGGGCGCAG", 4), ("GGCACGGGGCGCAGGGCGCGGGGCGCAGGGC", 1)],
        87,
    ),
    (
        "NC_000017.11:g.30221569_30221577CAGCCCAC[2]CCAGCCCAC[11]C[1]",
        [("CAGCCCAC", 2), ("CCAGCCCAC", 11), ("C", 1)],
        116,
    ),
]


def _change(hgvs: str):
    return parse(hgvs).changes[0]


@pytest.mark.parametrize(("hgvs", "segments", "length"), REAL_COMPOSITE)
def test_composite_repeat(hgvs, segments, length) -> None:
    change = _change(hgvs)
    assert change.kind is VariantKind.REPEAT
    assert [(s.unit, s.count) for s in change.repeat_segments] == segments
    assert change.repeat_length == length
    # Composite repeats have no single unit/count.
    assert change.repeat_unit is None
    assert change.repeat_count is None


def test_composite_repeat_range() -> None:
    change = _change("NC_000004.12:g.3074877_3074939CAG[19]CAA[4]")
    assert (change.start.base, change.end.base) == (3074877, 3074939)
    assert str(change.repeat_segments[0]) == "CAG[19]"


def test_composite_repeat_coding_and_lowercase() -> None:
    change = _change("NM_002111.8:c.54cag[19]caa[1]")
    assert [(s.unit, s.count) for s in change.repeat_segments] == [
        ("CAG", 19),
        ("CAA", 1),
    ]


def test_simple_repeat_keeps_unit_and_segment() -> None:
    change = _change("NM_002111.8:c.54CAG[23]")
    assert (change.repeat_unit, change.repeat_count) == ("CAG", 23)
    assert [(s.unit, s.count) for s in change.repeat_segments] == [("CAG", 23)]
    assert change.repeat_length == 69


def test_unitless_repeat() -> None:
    change = _change("NC_000001.11:g.100_102[7]")
    assert change.repeat_count == 7
    assert change.repeat_segments is None
    assert change.repeat_length is None


@pytest.mark.parametrize(
    "bad",
    [
        "NC_000001.11:g.100_102CAG[]",
        "NC_000001.11:g.100_102CAG[5]CAA",
        "NC_000001.11:g.100_102CAG[x]CAA[2]",
    ],
)
def test_malformed_repeat_rejected(bad: str) -> None:
    with pytest.raises(BioinformaticsError):
        parse(bad)
