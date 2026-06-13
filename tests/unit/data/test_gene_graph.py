"""Tests for src.data.library.gene_graph — per-gene variant-centric graph."""

from __future__ import annotations

import pytest

from src.data.library.consequence import CONSEQUENCE_DIM, consequence_vector
from src.data.library.gene_graph import (
    GENE_EDGE_DIM,
    GENE_NODE_DIM,
    GENE_NODE_STRUCT_DIM,
    build_gene_graph,
)
from src.data.library.ingest.models import IngestedHaplotype, IngestedVariant
from src.data.library.protein_change import PROTEIN_CHANGE_DIM, protein_change_vector
from src.genomics.annotation import GeneModel

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


_SUB = _v(
    150,
    "150C>T",
    ref="C",
    alt="T",
    so_terms=("missense_variant",),
    p_hgvs="NP_1:p.Pro296Ser",
)
_DEL = _v(250, "250del")
_INS = _v(350, "350_351insA", alt="A")


def test_schema_dims() -> None:
    assert GENE_NODE_DIM == GENE_NODE_STRUCT_DIM + CONSEQUENCE_DIM + PROTEIN_CHANGE_DIM
    assert GENE_NODE_DIM == 30
    assert GENE_EDGE_DIM == 2


def test_node_count_and_order() -> None:
    g = build_gene_graph(_GENE, variants=[_INS, _SUB, _DEL])
    assert g.x.shape == (4, GENE_NODE_DIM)  # anchor + 3 variants
    # Node 0 is the anchor; variants follow sorted by genomic position.
    assert g.node_pos == [100, 150, 250, 350]
    assert g.node_hgvs[0] == ""


def test_anchor_features() -> None:
    g = build_gene_graph(_GENE, variants=[_SUB])
    anchor = g.x[0].tolist()
    assert anchor[0] == 1.0  # is_anchor
    assert sum(anchor[1:]) == 0.0  # everything else zero


def test_variant_kind_one_hot() -> None:
    g = build_gene_graph(_GENE, variants=[_SUB, _DEL, _INS])
    # struct one-hot kinds occupy indices 2..6 (sub, del, ins, delins, other).
    assert g.x[1][2].item() == 1.0  # substitution
    assert g.x[2][3].item() == 1.0  # deletion
    assert g.x[3][4].item() == 1.0  # insertion


def test_consequence_and_protein_blocks() -> None:
    g = build_gene_graph(_GENE, variants=[_SUB])
    node = g.x[1].tolist()
    cons = node[GENE_NODE_STRUCT_DIM : GENE_NODE_STRUCT_DIM + CONSEQUENCE_DIM]
    prot = node[GENE_NODE_STRUCT_DIM + CONSEQUENCE_DIM :]
    assert cons == pytest.approx(consequence_vector("missense_variant"))
    assert prot == pytest.approx(protein_change_vector("NP_1:p.Pro296Ser"))


def test_paths_record_membership() -> None:
    haplos = [
        IngestedHaplotype(gene="GENEX", label="*1"),  # reference
        IngestedHaplotype(gene="GENEX", label="*4", variants=(_SUB, _INS)),
        IngestedHaplotype(gene="GENEX", label="*9", variants=(_DEL,)),
    ]
    g = build_gene_graph(_GENE, haplotypes=haplos)
    # Variants come only from the haplotypes; 3 distinct + anchor.
    assert g.x.shape[0] == 4
    assert g.paths["*1"] == ()
    # *4 holds the substitution (pos 150 → node 1) and insertion (pos 350 → node 3).
    assert g.paths["*4"] == (1, 3)
    assert g.paths["*9"] == (2,)


def test_edges_bidirectional_chain() -> None:
    g = build_gene_graph(_GENE, variants=[_SUB, _DEL])
    # 3 nodes → 2 backbone links → 4 directed edges.
    assert g.edge_index.shape == (2, 4)
    assert g.edge_attr.shape == (4, GENE_EDGE_DIM)
    # First link touches the anchor.
    assert g.edge_attr[0][0].item() == 1.0


def test_empty_gene_has_only_anchor() -> None:
    g = build_gene_graph(_GENE)
    assert g.x.shape == (1, GENE_NODE_DIM)
    assert g.edge_index.shape == (2, 0)
