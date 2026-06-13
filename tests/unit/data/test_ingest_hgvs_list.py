"""Tests for src.data.library.ingest.hgvs_list — raw genomic-HGVS ingestion."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.data.library.ingest.hgvs_list import (
    iter_variants_from_hgvs,
    variant_from_genomic_hgvs,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_variant_from_genomic_substitution() -> None:
    v = variant_from_genomic_hgvs("NC_000022.11:g.42128945C>T", gene="CYP2D6")
    assert v.accession == "NC_000022.11"
    assert v.pos == 42128945
    assert (v.ref, v.alt) == ("C", "T")
    assert v.gene == "CYP2D6"


def test_plain_one_per_line(tmp_path: Path) -> None:
    path = tmp_path / "variants.hgvs"
    path.write_text(
        "# a comment\nNC_000022.11:g.42128945C>T\n\nNC_000022.11:g.42130692G>A\n",
        encoding="utf-8",
    )
    variants = list(iter_variants_from_hgvs(path))
    assert [v.pos for v in variants] == [42128945, 42130692]
    assert all(v.gene is None for v in variants)


def test_tsv_with_gene_column(tmp_path: Path) -> None:
    path = tmp_path / "variants.tsv"
    path.write_text(
        "gene\thgvs\nCYP2D6\tNC_000022.11:g.42128945C>T\n",
        encoding="utf-8",
    )
    variants = list(iter_variants_from_hgvs(path))
    assert len(variants) == 1
    assert variants[0].gene == "CYP2D6"


def test_non_genomic_lines_skipped(tmp_path: Path) -> None:
    path = tmp_path / "mixed.hgvs"
    path.write_text(
        "NC_000022.11:g.42128945C>T\nNM_000106.6:c.886C>T\np.Pro296Ser\n",
        encoding="utf-8",
    )
    variants = list(iter_variants_from_hgvs(path))
    # Only the genomic line survives; c./p. are skipped (logged).
    assert len(variants) == 1
    assert variants[0].pos == 42128945
