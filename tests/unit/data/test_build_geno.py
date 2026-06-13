"""Tests for src.data.library.build_geno — library orchestration."""

from __future__ import annotations

from src.data.library.build_geno import build_geno_library
from src.data.library.ingest.models import IngestedHaplotype, IngestedVariant
from src.genomics.annotation import GeneAnnotation, GeneModel


def _model(symbol: str, accession: str) -> GeneModel:
    return GeneModel(
        symbol=symbol,
        accession=accession,
        start=100,
        stop=400,
        strand="+",
        biotype="protein_coding",
        transcripts=(),
    )


def _v(gene: str, accession: str, pos: int) -> IngestedVariant:
    return IngestedVariant(
        gene=gene,
        g_hgvs=f"{accession}:g.{pos}C>T",
        accession=accession,
        pos=pos,
        ref="C",
        alt="T",
    )


def test_builds_annotated_genes_skips_others() -> None:
    annotation = GeneAnnotation(
        {"GENEA": _model("GENEA", "NC_1"), "GENEB": _model("GENEB", "NC_2")}
    )
    variants = [
        _v("GENEA", "NC_1", 150),
        _v("GENEB", "NC_2", 150),
        _v("GENEC", "NC_3", 150),  # no annotation → skipped
    ]
    haplotypes = [
        IngestedHaplotype(
            gene="GENEA", label="*2", variants=(_v("GENEA", "NC_1", 150),)
        )
    ]
    library = build_geno_library(annotation, variants=variants, haplotypes=haplotypes)

    assert set(library.genes) == {"GENEA", "GENEB"}  # GENEC skipped
    assert "*2" in library.labels("GENEA")
    assert library.get("GENEA").x.shape[0] == 2  # anchor + 1 variant


def test_empty_inputs_give_empty_library() -> None:
    library = build_geno_library(GeneAnnotation({}))
    assert len(library) == 0
