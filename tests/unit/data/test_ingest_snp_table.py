"""Tests for src.data.library.ingest.snp_table — pan-gene SNP table adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.data.library.ingest.snp_table import (
    iter_variants_from_snp_table,
    rsid_hgvs_index,
)

if TYPE_CHECKING:
    from pathlib import Path

_TSV_WITH_RSID = "\n".join(
    [
        "snp\tgene\tchr\tstart_pos\tend_pos\tRef_Allele\tAlt_Allele\tFXN_CLASS",
        "rs100\tGENEA\t22\t100\t100\tC\tT\tmissense_variant",
        "rs200\tGENEA\t22\t200\t200\tAC\tA\tintron_variant",
        "\tGENEA\t22\t300\t300\tG\tA\t",  # no rsID → skipped
        "rs500\tGENEB\t7\t500\t500\t\t\t",  # blank alleles → skipped
        "",
    ]
)

_TSV = "\n".join(
    [
        "gene\tchr\tstart_pos\tend_pos\tRef_Allele\tAlt_Allele\tFXN_CLASS",
        "GENEA\t22\t100\t100\tC\tT\tmissense_variant,coding_sequence_variant",
        "GENEA\t22\t200\t200\tAC\tA\tintron_variant",
        "GENEB\t7\t500\t500\t\t\t",  # blank alleles → skipped
        "",
    ]
)


def test_parses_variants_with_so_terms(tmp_path: Path) -> None:
    path = tmp_path / "snp_data_output.tsv"
    path.write_text(_TSV, encoding="utf-8")
    variants = list(iter_variants_from_snp_table(path))

    assert len(variants) == 2  # the blank-allele row is skipped
    first = variants[0]
    assert first.gene == "GENEA"
    assert first.g_hgvs == "NC_000022.11:g.100C>T"
    assert first.so_terms == ("missense_variant", "coding_sequence_variant")
    # AC>A is a VCF-style deletion → trimmed to g.201del.
    assert variants[1].g_hgvs == "NC_000022.11:g.201del"


def test_rsid_hgvs_index(tmp_path: Path) -> None:
    path = tmp_path / "snp_data_output.tsv"
    path.write_text(_TSV_WITH_RSID, encoding="utf-8")
    index = rsid_hgvs_index(path)

    assert index == {
        "rs100": "NC_000022.11:g.100C>T",
        "rs200": "NC_000022.11:g.201del",
    }


def test_rsid_index_missing_snp_column_is_empty(tmp_path: Path) -> None:
    path = tmp_path / "snp_data_output.tsv"
    path.write_text(_TSV, encoding="utf-8")  # the no-snp fixture
    assert rsid_hgvs_index(path) == {}
