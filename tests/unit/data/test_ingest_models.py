"""Tests for src.data.library.ingest.models — shared ingestion shapes."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.data.library.ingest.models import IngestedHaplotype, IngestedVariant


def _variant(**over: object) -> IngestedVariant:
    base: dict[str, object] = {
        "gene": "CYP2D6",
        "g_hgvs": "NC_000022.11:g.42128945C>T",
        "accession": "NC_000022.11",
        "pos": 42128945,
        "ref": "c",
        "alt": "t",
    }
    base.update(over)
    return IngestedVariant(**base)


def test_alleles_uppercased() -> None:
    v = _variant()
    assert v.ref == "C" and v.alt == "T"


def test_defaults_empty() -> None:
    v = _variant()
    assert v.c_hgvs is None and v.p_hgvs is None and v.so_terms == ()


def test_frozen() -> None:
    v = _variant()
    with pytest.raises(ValidationError):
        v.ref = "G"  # type: ignore[misc]


def test_pos_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        _variant(pos=0)


def test_haplotype_groups_variants() -> None:
    haplo = IngestedHaplotype(
        gene="CYP2D6",
        label="*4",
        variants=(
            _variant(),
            _variant(
                pos=42129809, g_hgvs="NC_000022.11:g.42129809G>A", ref="G", alt="A"
            ),
        ),
    )
    assert haplo.label == "*4"
    assert len(haplo.variants) == 2
    assert {v.pos for v in haplo.variants} == {42128945, 42129809}


def test_empty_haplotype_is_reference() -> None:
    assert IngestedHaplotype(gene="CYP2D6", label="*1").variants == ()
