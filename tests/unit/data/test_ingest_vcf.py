"""Tests for src.data.library.ingest.vcf — sample VCF → genomic-HGVS variants.

``iter_variants`` (pysam, FASTA-backed) is stubbed so the adapter's mapping logic
is tested without real VCF/FASTA fixtures.
"""

from __future__ import annotations

from pathlib import Path

from src.data.library.ingest.vcf import iter_variants_from_vcf
from src.genomics import variant_val


def _fake_iter_variants(vcf_path, fasta_path, region=None):  # noqa: ANN001, ARG001
    yield {"chrom": "22", "pos": 42128945, "ref": "C", "alts": ("T",)}
    yield {"chrom": "22", "pos": 100, "ref": "AC", "alts": ("A",)}  # deletion
    yield {"chrom": "22", "pos": 200, "ref": "G", "alts": ("GA", "GT")}  # multi-allelic


def test_maps_records_to_genomic_hgvs(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(variant_val, "iter_variants", _fake_iter_variants)
    variants = list(
        iter_variants_from_vcf(Path("sample.vcf"), Path("ref.fa"), gene="CYP2D6")
    )

    g = [v.g_hgvs for v in variants]
    assert g == [
        "NC_000022.11:g.42128945C>T",  # SNV
        "NC_000022.11:g.101del",  # AC>A trimmed
        "NC_000022.11:g.200_201insA",  # G>GA insertion
        "NC_000022.11:g.200_201insT",  # G>GT insertion (second ALT)
    ]
    assert all(v.gene == "CYP2D6" for v in variants)
    assert all(v.accession == "NC_000022.11" for v in variants)


def test_substitution_anchor(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(variant_val, "iter_variants", _fake_iter_variants)
    first = next(iter(iter_variants_from_vcf(Path("s.vcf"), Path("r.fa"))))
    assert (first.pos, first.ref, first.alt) == (42128945, "C", "T")
