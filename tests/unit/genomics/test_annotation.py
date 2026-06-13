"""Tests for src.genomics.annotation — RefSeq GFF3 gene-model reader."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.genomics.annotation import GeneAnnotation, gene_sequence

if TYPE_CHECKING:
    from pathlib import Path

# A minimal RefSeq-style GFF3: two genes on one contig, GENEA (+) coding with two
# exons / one CDS, GENEB (-) single-exon. Tabs are significant.
_GFF = "\n".join(
    [
        "#gff-version 3",
        "NC_TEST.1\tRefSeq\tregion\t1\t60\t.\t+\t.\tID=NC_TEST.1:1..60",
        "NC_TEST.1\tBestRefSeq\tgene\t11\t40\t.\t+\t.\tID=gene-GENEA;Name=GENEA;gene=GENEA;gene_biotype=protein_coding",
        "NC_TEST.1\tBestRefSeq\tmRNA\t11\t40\t.\t+\t.\tID=rna-NM_TEST.1;Parent=gene-GENEA;Name=NM_TEST.1",
        "NC_TEST.1\tBestRefSeq\texon\t11\t20\t.\t+\t.\tID=exon-1;Parent=rna-NM_TEST.1",
        "NC_TEST.1\tBestRefSeq\texon\t31\t40\t.\t+\t.\tID=exon-2;Parent=rna-NM_TEST.1",
        "NC_TEST.1\tBestRefSeq\tCDS\t12\t20\t.\t+\t0\tID=cds-NP_TEST.1;Parent=rna-NM_TEST.1;Name=NP_TEST.1",
        "NC_TEST.1\tBestRefSeq\tCDS\t31\t39\t.\t+\t0\tID=cds-NP_TEST.1;Parent=rna-NM_TEST.1;Name=NP_TEST.1",
        "NC_TEST.1\tBestRefSeq\tgene\t45\t54\t.\t-\t.\tID=gene-GENEB;Name=GENEB;gene=GENEB;gene_biotype=protein_coding",
        "NC_TEST.1\tBestRefSeq\tmRNA\t45\t54\t.\t-\t.\tID=rna-NM_TEST.2;Parent=gene-GENEB;Name=NM_TEST.2",
        "NC_TEST.1\tBestRefSeq\texon\t45\t54\t.\t-\t.\tID=exon-3;Parent=rna-NM_TEST.2",
        "",
    ]
)

_SEQ = "ACGT" * 15  # 60 bp


@pytest.fixture
def gff_path(tmp_path: Path) -> Path:
    path = tmp_path / "test.gff"
    path.write_text(_GFF, encoding="utf-8")
    return path


@pytest.fixture
def fasta_path(tmp_path: Path) -> Path:
    path = tmp_path / "test.fa"
    path.write_text(f">NC_TEST.1 test contig\n{_SEQ}\n", encoding="utf-8")
    return path


class TestFromGff:
    def test_parses_both_genes(self, gff_path: Path) -> None:
        ann = GeneAnnotation.from_gff(gff_path)
        assert len(ann) == 2
        assert "GENEA" in ann and "GENEB" in ann

    def test_gene_fields(self, gff_path: Path) -> None:
        gene = GeneAnnotation.from_gff(gff_path).get("GENEA")
        assert gene is not None
        assert gene.accession == "NC_TEST.1"
        assert (gene.start, gene.stop, gene.strand) == (11, 40, "+")
        assert gene.biotype == "protein_coding"
        assert gene.length == 30

    def test_transcript_structure(self, gff_path: Path) -> None:
        gene = GeneAnnotation.from_gff(gff_path).get("GENEA")
        assert gene is not None and len(gene.transcripts) == 1
        tx = gene.transcripts[0]
        assert tx.accession == "NM_TEST.1"
        assert tx.protein_accession == "NP_TEST.1"
        assert tx.exons == ((11, 20), (31, 40))
        assert tx.cds == ((12, 20), (31, 39))

    def test_minus_strand_gene(self, gff_path: Path) -> None:
        gene = GeneAnnotation.from_gff(gff_path).get("GENEB")
        assert gene is not None and gene.strand == "-"
        # Non-coding-side transcript: exon present, no CDS / protein.
        assert gene.transcripts[0].cds == ()
        assert gene.transcripts[0].protein_accession is None

    def test_genes_filter_skips_others(self, gff_path: Path) -> None:
        ann = GeneAnnotation.from_gff(gff_path, genes=["GENEA"])
        assert "GENEA" in ann and "GENEB" not in ann
        assert len(ann) == 1


def test_gene_sequence_plus_strand(gff_path: Path, fasta_path: Path) -> None:
    pyfaidx = pytest.importorskip("pyfaidx")
    gene = GeneAnnotation.from_gff(gff_path).get("GENEA")
    assert gene is not None
    fasta = pyfaidx.Fasta(str(fasta_path), key_function=lambda x: x.split()[0])
    try:
        seq = gene_sequence(gene, fasta)
    finally:
        fasta.close()
    # 1-based inclusive [11, 40] → 0-based slice [10:40].
    assert seq == _SEQ[10:40]
    assert len(seq) == gene.length
