"""Per-gene variation graph builder (variant-centric topology).

One PyG graph per gene. Node 0 is the reference (``*1``) anchor; nodes ``1..N`` are
the gene's distinct variants, ordered by genomic position. Edges chain consecutive
nodes (anchor → v1 → … → vN) so message passing follows genomic order. Each star
allele / haplotype is recorded as a **path** — the indices of its variant nodes —
so a genotype is one path and a diplotype is the union of two; the path subgraph is
materialised at access time (see the dataset layer).

Node features (``GENE_NODE_DIM`` = 30):
    structural (9)  [is_anchor, is_variant, kind∈{sub,del,ins,delins,other}(5),
                     position_norm, length_norm]
    consequence (13) Sequence Ontology multi-hot + severity + known (Layer B,
                     :mod:`src.data.library.consequence`), per variant.
    protein (8)      HGVS protein-change physicochemistry (Layer C,
                     :mod:`src.data.library.protein_change`), per variant.

The per-allele PGx **function** (no/decreased/normal/increased + activity) is a
property of the *path*, not a node, and is attached separately (next step).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch_geometric.data.data import Data

from src.data.library.consequence import CONSEQUENCE_DIM, consequence_vector
from src.data.library.haplotype_function import HaplotypeFunctionProvider
from src.data.library.protein_change import PROTEIN_CHANGE_DIM, protein_change_vector

if TYPE_CHECKING:
    from collections.abc import Iterable

    from src.data.library.ingest.models import IngestedHaplotype, IngestedVariant
    from src.genomics.annotation import GeneModel

logger = logging.getLogger(__name__)

# --- Schema (sources of truth synced in the model wiring step) --------------- #
GENE_NODE_STRUCT_DIM: int = 9
GENE_NODE_DIM: int = GENE_NODE_STRUCT_DIM + CONSEQUENCE_DIM + PROTEIN_CHANGE_DIM  # 30
GENE_EDGE_DIM: int = 2

# Variant-kind one-hot order within the structural block.
_KINDS: tuple[str, ...] = ("substitution", "deletion", "insertion", "delins", "other")
_KIND_INDEX: dict[str, int] = {k: i for i, k in enumerate(_KINDS)}
_LENGTH_CAP: float = 50.0  # bp; change sizes saturate here for length_norm


def _kind_and_size(g_hgvs: str) -> tuple[str, int]:
    """Classify a variant and estimate its change size from its ``g.`` body.

    String-based (no re-parse): the body suffix is unambiguous — ``delins`` before
    ``del``/``ins``, then a substitution ``>``. Size is the affected span for
    del/delins, the inserted length for ins, else 1.
    """
    body = g_hgvs.rsplit(":", maxsplit=1)[-1]
    if "delins" in body:
        return "delins", _span_size(body)
    if "ins" in body:
        _, _, inserted = body.partition("ins")
        return "insertion", max(len(inserted), 1)
    if "del" in body:
        return "deletion", _span_size(body)
    if ">" in body:
        return "substitution", 1
    return "other", 1


def _span_size(body: str) -> int:
    """Span length from a ``g.<start>_<stop>...`` body (1 when single-position)."""
    # "del" prefixes "delins", so a single split truncates either suffix.
    coords = body[2:].split("del", maxsplit=1)[0]
    if "_" in coords:
        start_s, _, stop_s = coords.partition("_")
        try:
            return abs(int(stop_s) - int(start_s)) + 1
        except ValueError:
            return 1
    return 1


def _struct_block(*, is_anchor: bool, g_hgvs: str, position_norm: float) -> list[float]:
    vec = [0.0] * GENE_NODE_STRUCT_DIM
    if is_anchor:
        vec[0] = 1.0
        return vec
    vec[1] = 1.0
    kind, size = _kind_and_size(g_hgvs)
    vec[2 + _KIND_INDEX[kind]] = 1.0
    vec[7] = max(0.0, min(position_norm, 1.0))
    vec[8] = min(size / _LENGTH_CAP, 1.0)
    return vec


def _node_features(variant: IngestedVariant, position_norm: float) -> list[float]:
    """Full node vector for a variant: structural + consequence + protein."""
    struct = _struct_block(
        is_anchor=False, g_hgvs=variant.g_hgvs, position_norm=position_norm
    )
    consequence = consequence_vector(",".join(variant.so_terms) or None)
    protein = protein_change_vector(variant.p_hgvs)
    return [*struct, *consequence, *protein]


def _collect_variants(
    variants: Iterable[IngestedVariant],
    haplotypes: Iterable[IngestedHaplotype],
) -> list[IngestedVariant]:
    """Distinct variants (by ``g_hgvs``) from loose variants + all haplotypes."""
    seen: dict[str, IngestedVariant] = {}
    for source in (variants, *(h.variants for h in haplotypes)):
        for v in source:
            seen.setdefault(v.g_hgvs, v)
    return sorted(seen.values(), key=lambda v: v.pos)


def build_gene_graph(
    gene: GeneModel,
    variants: Iterable[IngestedVariant] = (),
    haplotypes: Iterable[IngestedHaplotype] = (),
    *,
    function_provider: HaplotypeFunctionProvider | None = None,
) -> Data:
    """Build the variant-centric variation graph for one gene.

    ``variants`` are standalone variants (VCF / HGVS list); ``haplotypes`` are
    named alleles whose variants also populate the graph and whose membership is
    recorded in ``data.paths`` (a reference allele maps to an empty path).
    ``function_provider`` supplies each path's PGx function vector
    (``data.path_function``); a null provider leaves them zero.
    """
    haplotypes = list(haplotypes)
    functions = function_provider or HaplotypeFunctionProvider.null()
    ordered = _collect_variants(variants, haplotypes)

    span = max(gene.length - 1, 1)
    # Node 0 = reference anchor; nodes 1..N = variants in genomic order.
    x: list[list[float]] = [
        _struct_block(is_anchor=True, g_hgvs="", position_norm=0.0)
        + [0.0] * (CONSEQUENCE_DIM + PROTEIN_CHANGE_DIM)
    ]
    node_pos: list[int] = [gene.start]
    node_hgvs: list[str] = [""]
    index_of: dict[str, int] = {}
    for v in ordered:
        idx = len(x)
        index_of[v.g_hgvs] = idx
        x.append(_node_features(v, (v.pos - gene.start) / span))
        node_pos.append(v.pos)
        node_hgvs.append(v.g_hgvs)

    edge_index, edge_attr = chain_edges(node_pos, gene.length)

    paths = {
        h.label: tuple(index_of[v.g_hgvs] for v in h.variants if v.g_hgvs in index_of)
        for h in haplotypes
    }

    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
    data.gene = gene.symbol
    data.accession = gene.accession
    data.gene_length = gene.length
    data.node_pos = node_pos
    data.node_hgvs = node_hgvs
    data.paths = paths
    data.path_function = {
        label: functions.vector_for(gene.symbol, label) for label in paths
    }
    logger.debug(
        "Gene graph %s: %d variants, %d paths.", gene.symbol, len(ordered), len(paths)
    )
    return data


def chain_edges(
    node_pos: list[int], gene_length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bidirectional chain over nodes in order; edge_attr = [is_anchor_link, gap].

    ``node_pos`` is the genomic position of each node (anchor first); reused both
    for the full gene graph and for path/diplotype subgraphs.
    """
    if len(node_pos) < 2:
        return (
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0, GENE_EDGE_DIM), dtype=torch.float32),
        )
    src: list[int] = []
    dst: list[int] = []
    attr: list[list[float]] = []
    span = max(gene_length, 1)
    for i in range(len(node_pos) - 1):
        gap = abs(node_pos[i + 1] - node_pos[i]) / span
        is_anchor_link = 1.0 if i == 0 else 0.0
        for a, b in ((i, i + 1), (i + 1, i)):
            src.append(a)
            dst.append(b)
            attr.append([is_anchor_link, min(gap, 1.0)])
    return (
        torch.tensor([src, dst], dtype=torch.long),
        torch.tensor(attr, dtype=torch.float32),
    )


__all__ = [
    "GENE_EDGE_DIM",
    "GENE_NODE_DIM",
    "GENE_NODE_STRUCT_DIM",
    "build_gene_graph",
    "chain_edges",
]
