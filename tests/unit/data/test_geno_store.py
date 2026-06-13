"""Tests for src.data.library.geno_store — single-file store + subgraph encode."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.data.library.gene_graph import GENE_NODE_DIM, build_gene_graph
from src.data.library.geno_store import GenoLibrary
from src.data.library.haplotype_function import (
    PATH_FUNCTION_DIM,
    HaplotypeFunctionProvider,
)
from src.data.library.ingest.models import IngestedHaplotype, IngestedVariant
from src.genomics.annotation import GeneModel

if TYPE_CHECKING:
    from pathlib import Path

_GENE = GeneModel(
    symbol="GENEX",
    accession="NC_000022.11",
    start=100,
    stop=400,
    strand="+",
    biotype="protein_coding",
    transcripts=(),
)


def _v(pos: int, body: str, **over: object) -> IngestedVariant:
    base: dict[str, object] = {
        "gene": "GENEX",
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
_INS = _v(350, "350_351insA", alt="A")


@pytest.fixture
def library() -> GenoLibrary:
    haplos = [
        IngestedHaplotype(gene="GENEX", label="*1"),
        IngestedHaplotype(gene="GENEX", label="*4", variants=(_SUB, _INS)),
        IngestedHaplotype(gene="GENEX", label="*9", variants=(_DEL,)),
    ]
    fn = HaplotypeFunctionProvider({("GENEX", "4"): ("no_function", 0.0)})
    graph = build_gene_graph(_GENE, haplotypes=haplos, function_provider=fn)
    return GenoLibrary({"GENEX": graph})


def test_basic_accessors(library: GenoLibrary) -> None:
    assert "GENEX" in library and len(library) == 1
    assert set(library.labels("GENEX")) == {"*1", "*4", "*9"}


def test_save_load_roundtrip(library: GenoLibrary, tmp_path: Path) -> None:
    path = tmp_path / "geno_graphs.pt"
    library.save(path)
    reloaded = GenoLibrary.load(path)
    assert reloaded.genes == ["GENEX"]
    assert (
        reloaded.encode("GENEX", "*4").x.shape == library.encode("GENEX", "*4").x.shape
    )


def test_encode_single_haplotype(library: GenoLibrary) -> None:
    data = library.encode("GENEX", "*4")
    # anchor + 2 variant nodes of *4.
    assert data.x.shape == (3, GENE_NODE_DIM)
    assert data.gene == "GENEX" and data.labels == ["*4"]
    assert data.geno_function.shape == (1, PATH_FUNCTION_DIM)
    assert data.geno_function[0][0].item() == 1.0  # no_function carried through


def test_encode_diplotype_is_union(library: GenoLibrary) -> None:
    data = library.encode("GENEX", ["*4", "*9"])
    # anchor + {*4: 2 nodes} ∪ {*9: 1 node} = 4 nodes.
    assert data.x.shape == (4, GENE_NODE_DIM)
    assert data.labels == ["*4", "*9"]


def test_encode_reference_is_anchor_only(library: GenoLibrary) -> None:
    data = library.encode("GENEX", "*1")
    assert data.x.shape == (1, GENE_NODE_DIM)
    assert data.edge_index.shape == (2, 0)


def test_encode_missing_gene_raises(library: GenoLibrary) -> None:
    with pytest.raises(KeyError, match="not in GenoLibrary"):
        library.encode("NOPE", "*1")
