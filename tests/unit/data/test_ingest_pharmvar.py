"""Tests for src.data.library.ingest.pharmvar — star allele = grouped HGVS variants."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.data.library.ingest.pharmvar import (
    iter_haplotypes,
    load_gene_haplotypes,
    rsid_hgvs_index,
)

if TYPE_CHECKING:
    from pathlib import Path

# A minimal PharmVar haplotypes TSV: a reference allele, a single-variant allele,
# and a multi-variant sub-allele spanning a substitution, deletion and insertion.
_TSV = "\n".join(
    [
        "#version=pharmvar-test",
        "Haplotype Name\tGene\trsID\tReferenceSequence\tVariant Start\tVariant Stop\tReference Allele\tVariant Allele\tType",
        "GENEX*1\tGENEX\t\tREFERENCE\t.\t\t\t\t",
        "GENEX*2\tGENEX\trs1\tNC_000022.11\t100\t100\tC\tT\tsubstitution",
        "GENEX*2.001\tGENEX\trs1\tNC_000022.11\t100\t100\tC\tT\tsubstitution",
        "GENEX*2.001\tGENEX\trs2\tNC_000022.11\t200\t200\tG\t-\tdeletion",
        "GENEX*2.001\tGENEX\trs3\tNC_000022.11\t300\t301\t-\tAC\tinsertion",
        "",
    ]
)


@pytest.fixture
def tsv_path(tmp_path: Path) -> Path:
    path = tmp_path / "GENEX.NC_000022.11.haplotypes.tsv"
    path.write_text(_TSV, encoding="utf-8")
    return path


def _by_label(haplos: list) -> dict[str, object]:
    return {h.label: h for h in haplos}


def test_groups_into_three_haplotypes(tsv_path: Path) -> None:
    haplos = _by_label(load_gene_haplotypes(tsv_path))
    assert set(haplos) == {"*1", "*2", "*2.001"}


def test_reference_allele_has_no_variants(tsv_path: Path) -> None:
    haplos = _by_label(load_gene_haplotypes(tsv_path))
    assert haplos["*1"].variants == ()


def test_single_variant_allele(tsv_path: Path) -> None:
    haplos = _by_label(load_gene_haplotypes(tsv_path))
    v = haplos["*2"].variants
    assert len(v) == 1
    assert v[0].g_hgvs == "NC_000022.11:g.100C>T"
    assert v[0].gene == "GENEX"


def test_multi_variant_sub_allele_sorted_with_indels(tsv_path: Path) -> None:
    haplos = _by_label(load_gene_haplotypes(tsv_path))
    variants = haplos["*2.001"].variants
    assert [v.pos for v in variants] == [100, 200, 300]  # sorted by position
    assert [v.g_hgvs for v in variants] == [
        "NC_000022.11:g.100C>T",
        "NC_000022.11:g.200del",
        "NC_000022.11:g.300_301insAC",
    ]


def test_iter_haplotypes_scans_folder(tsv_path: Path) -> None:
    # tsv_path lives in tmp_path; scanning the parent finds it.
    labels = {h.label for h in iter_haplotypes(tsv_path.parent)}
    assert labels == {"*1", "*2", "*2.001"}


def test_missing_folder_yields_nothing(tmp_path: Path) -> None:
    assert list(iter_haplotypes(tmp_path / "nope")) == []


def test_rsid_hgvs_index(tsv_path: Path) -> None:
    index = rsid_hgvs_index(tsv_path.parent)
    assert index == {
        "rs1": "NC_000022.11:g.100C>T",
        "rs2": "NC_000022.11:g.200del",
        "rs3": "NC_000022.11:g.300_301insAC",
    }
