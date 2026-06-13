"""Tests for src.data.library.genotype_resolver — genotype → encoded subgraph."""

from __future__ import annotations

import pytest

from src.data.library.gene_graph import GENE_NODE_DIM, build_gene_graph
from src.data.library.geno_store import GenoLibrary
from src.data.library.genotype_resolver import GenotypeResolver
from src.data.library.haplotype_function import HaplotypeFunctionProvider
from src.data.library.ingest.models import IngestedHaplotype, IngestedVariant
from src.genomics.annotation import GeneModel

_GENE = GeneModel(
    symbol="CYP2D6",
    accession="NC_000022.11",
    start=100,
    stop=400,
    strand="+",
    biotype="protein_coding",
    transcripts=(),
)


def _v(pos: int, body: str, **over: object) -> IngestedVariant:
    base: dict[str, object] = {
        "gene": "CYP2D6",
        "g_hgvs": f"NC_000022.11:g.{body}",
        "accession": "NC_000022.11",
        "pos": pos,
        "ref": "",
        "alt": "",
    }
    base.update(over)
    return IngestedVariant(**base)


_SUB = _v(150, "150C>T", ref="C", alt="T")
_DEL = _v(250, "250del")


@pytest.fixture
def resolver() -> GenotypeResolver:
    haplos = [
        IngestedHaplotype(gene="CYP2D6", label="*1"),
        IngestedHaplotype(gene="CYP2D6", label="*4", variants=(_SUB,)),
    ]
    fn = HaplotypeFunctionProvider({("CYP2D6", "4"): ("no_function", 0.0)})
    # _DEL is a standalone variant (no haplotype uses it) so both rsIDs resolve.
    graph = build_gene_graph(
        _GENE, variants=(_DEL,), haplotypes=haplos, function_provider=fn
    )
    library = GenoLibrary({"CYP2D6": graph})
    rsid_to_hgvs = {
        "rs3892097": "NC_000022.11:g.150C>T",
        "rs5030655": "NC_000022.11:g.250del",
    }
    return GenotypeResolver(library, rsid_to_hgvs)


def test_resolves_rsids_to_variant_path(resolver: GenotypeResolver) -> None:
    data = resolver.resolve("CYP2D6", "rs3892097|rs5030655")
    assert data is not None
    assert data.x.shape == (3, GENE_NODE_DIM)  # anchor + 2 variants
    assert data.labels == []  # ad-hoc path


def test_star_allele_token_takes_label_route(resolver: GenotypeResolver) -> None:
    data = resolver.resolve("CYP2D6", "*4")
    assert data is not None
    assert data.labels == ["*4"]
    assert data.geno_function[0][0].item() == 1.0  # function preserved via label


def test_unknown_rsid_dropped(resolver: GenotypeResolver) -> None:
    data = resolver.resolve("CYP2D6", "rs3892097|rs999999")
    assert data is not None
    assert data.x.shape == (2, GENE_NODE_DIM)  # anchor + the one known variant


def test_gene_not_in_library_returns_none(resolver: GenotypeResolver) -> None:
    assert resolver.resolve("UGT1A1", "rs3892097") is None
