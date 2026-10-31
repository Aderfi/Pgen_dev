"""Genotype → HGVS variant graph (the input of the genotype tower).

A patient's genotype is read as a set of HGVS variants around the gene(s) it
concerns::

        gene hub ──── v1 ── v2 ── v3        (variants in genomic order)
            ╲──────────╱     ╱
             ╲──────────────╱

* **Gene nodes** (one per gene of the row) carry the gene's identity as a
  learned embedding (``gene_token``) — a missense change means different things
  in CYP2D6 and in XRCC1.
* **Variant nodes** (one per elementary HGVS change) carry what the HGVS
  notation states plus catalog annotation and allele dosage
  (:mod:`src.data.genotype.features`).
* **Edges**: gene ↔ variant membership, and variant ↔ next variant on the same
  reference sequence (cis-neighbourhood; ``edge_attr`` holds the log genomic
  distance). Edge layout: ``[is_membership, is_adjacency, distance_norm]``.

Genotype tokens are linked to HGVS in this order (first wins per variant):
explicit HGVS tokens, the annotation's allele-specific variants (KG
``IMPLICATES``), rsID labels (narrowed to the alternative the ``alleles``
field names), and star alleles (``*4``, from ``genotype`` or ``alleles``,
expanded to their cis variants). Tokens that name no variant (``REF_SEQ``,
``Gene_Deletion``) add nothing; a genotype with no variants is the gene node(s)
alone — the reference.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch_geometric.data import Data

from src.data.genotype.features import (
    UNKNOWN_DOSAGE,
    Dosage,
    alt_allele,
    gene_node_vector,
    parse_hgvs,
    start_position,
    variant_node_vector,
)
from src.data.library.haplotype_function import PATH_FUNCTION_DIM

if TYPE_CHECKING:
    from collections.abc import Iterable

    from src.data.genotype.catalog import HgvsCatalog

logger = logging.getLogger(__name__)

GENO_EDGE_DIM = 3
PAD_TOKEN = 0  # variant nodes (no gene identity)
UNK_TOKEN = 1  # genes outside the training vocabulary
MULTI_VALUE_SEP = "|"

_NUCLEOTIDE_CALL = re.compile(r"^([ACGT])/?([ACGT])?$")
_RSID = re.compile(r"^rs\d+$")
_MAX_DISTANCE_LOG10 = 7.0  # ~10 Mb saturates the distance feature


def split_values(cell: object) -> list[str]:
    """``"A|B|A"`` → ``["A", "B"]`` (stripped, de-duplicated, order kept)."""
    if cell is None:
        return []
    return list(
        dict.fromkeys(t.strip() for t in str(cell).split(MULTI_VALUE_SEP) if t.strip())
    )


def build_gene_vocab(
    gene_cells: Iterable[object], min_count: int = 2
) -> dict[str, int]:
    """Gene symbol → embedding index for genes seen at least ``min_count`` times.

    Rarer genes share ``UNK_TOKEN``, so its embedding is trained too. Index 0
    is reserved for variant nodes.
    """
    counts = Counter(g for cell in gene_cells for g in split_values(cell))
    kept = sorted(g for g, n in counts.items() if n >= min_count)
    return {gene: i for i, gene in enumerate(kept, start=UNK_TOKEN + 1)}


def gene_vocab_size(vocab: dict[str, int]) -> int:
    """Rows needed in the gene embedding table for ``vocab``."""
    return max(vocab.values(), default=UNK_TOKEN) + 1


# --------------------------------------------------------------------------- #
# ``alleles`` field → what the patient carries
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AlleleCall:
    """The single genotype an ``alleles`` cell states (None when ambiguous)."""

    bases: tuple[str, ...] = ()  # nucleotide call, e.g. ("C", "T")
    stars: tuple[str, ...] = ()  # star diplotype, e.g. ("*4", "*5")

    @classmethod
    def parse(cls, alleles: object) -> AlleleCall | None:
        """``"CT"`` / ``"C/T"`` / ``"*4/*5"`` → a call; ``"CC + CT"`` → ``None``."""
        if alleles is None or not str(alleles).strip():
            return None
        groups = [g.strip() for g in str(alleles).split("+") if g.strip()]
        if len(groups) != 1:
            return None  # several genotypes grouped together — no single call
        text = groups[0].replace(" ", "")
        if m := _NUCLEOTIDE_CALL.match(text.upper()):
            return cls(bases=tuple(b for b in m.groups() if b))
        stars = tuple(t for t in text.split("/") if t.startswith("*"))
        return cls(stars=stars) if stars else None

    def copies_of_base(self, alt: str) -> int | None:
        """Alt copies when the call is a full diploid genotype (``"CT"``)."""
        if len(self.bases) != 2:
            return None
        return sum(b == alt for b in self.bases)


def _star_tokens(text: object) -> list[str]:
    """Star-allele tokens in a free-text cell (``"*1/*3 + *3/*3"`` → ``*1``, ``*3``)."""
    if text is None:
        return []
    return list(dict.fromkeys(re.findall(r"(?:[A-Z0-9-]+)?\*[\w.:]+", str(text))))


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _Linked:
    hgvs: str
    dosage: Dosage
    source: str


@dataclass
class _Node:
    features: list[float]
    reference: str
    position: int | None
    gene: str | None


class GenotypeGraphBuilder:
    """Builds the HGVS variant graph for one ``(gene, genotype)`` row.

    Args:
        catalog: Label/annotation → HGVS links and variant annotation.
        gene_vocab: Gene symbol → embedding index (see :func:`build_gene_vocab`).
    """

    def __init__(self, catalog: HgvsCatalog, gene_vocab: dict[str, int]) -> None:
        self.catalog = catalog
        self.gene_vocab = gene_vocab
        self.stats: Counter[str] = Counter()

    # ----- linking --------------------------------------------------------- #

    def _dosage(self, hgvs: str, call: AlleleCall | None) -> Dosage:
        parsed = parse_hgvs(hgvs)
        alt = alt_allele(parsed) if parsed is not None else None
        if call is None or alt is None:
            return UNKNOWN_DOSAGE
        return Dosage(call.copies_of_base(alt))

    def _rsid_variants(self, rsid: str, call: AlleleCall | None) -> list[str]:
        """The site's alternatives, narrowed to the one(s) the call observed."""
        candidates = list(self.catalog.labels.get(rsid, ()))
        if call is None or not call.bases:
            return candidates
        observed = set(call.bases)
        narrowed = []
        for hgvs in candidates:
            parsed = parse_hgvs(hgvs)
            if parsed is not None and alt_allele(parsed) in observed:
                narrowed.append(hgvs)
        return narrowed or candidates

    def _star_label(self, token: str, genes: list[str]) -> str | None:
        if not token.startswith("*"):
            return token
        for gene in genes:
            label = f"{gene}{token}"
            if label in self.catalog.labels:
                return label
        return f"{genes[0]}{token}" if genes else None

    def link(
        self,
        genes: list[str],
        genotype: object,
        alleles: object = None,
        annotation_id: object = None,
    ) -> tuple[list[_Linked], list[list[float]]]:
        """HGVS variants of the row (with dosage + source) and star-allele functions."""
        tokens = split_values(genotype)
        call = AlleleCall.parse(alleles)
        linked: dict[str, _Linked] = {}

        def add(hgvs: str, dosage: Dosage, source: str) -> None:
            linked.setdefault(hgvs, _Linked(hgvs, dosage, source))

        for token in tokens:
            if parse_hgvs(token) is not None:
                add(token, self._dosage(token, call), "hgvs")

        implicated = self.catalog.annotations.get(str(annotation_id), ())
        for hgvs in implicated:
            add(hgvs, self._dosage(hgvs, call), "annotation")

        if not implicated:
            for token in tokens:
                if _RSID.match(token):
                    for hgvs in self._rsid_variants(token, call):
                        add(hgvs, self._dosage(hgvs, call), "rsid")

        functions: list[list[float]] = []
        star_copies = Counter(call.stars) if call is not None else Counter()
        diplotype_known = call is not None and len(call.stars) == 2
        stars = [*(t for t in tokens if "*" in t), *_star_tokens(alleles)]
        for token in dict.fromkeys(stars):
            label = self._star_label(token, genes)
            if label is None:
                continue
            gene, _, allele = label.partition("*")
            functions.append(self.catalog.functions.vector_for(gene, f"*{allele}"))
            copies = star_copies.get(token) if diplotype_known else None
            for hgvs in self.catalog.labels.get(label, ()):
                add(hgvs, Dosage(copies), "star")
        return list(linked.values()), functions

    # ----- graph ----------------------------------------------------------- #

    def _variant_nodes(self, linked: list[_Linked], genes: list[str]) -> list[_Node]:
        nodes: list[_Node] = []
        for item in linked:
            parsed = parse_hgvs(item.hgvs)
            if parsed is None:
                self.stats["unparseable_hgvs"] += 1
                continue
            info = self.catalog.info(item.hgvs)
            gene = next(
                (g for g in (info.gene, parsed.gene_symbol) if g and g in genes), None
            )
            # A compound expression ([a;b]) contributes one node per change.
            for change in parsed.changes:
                nodes.append(
                    _Node(
                        features=variant_node_vector(
                            parsed,
                            change,
                            so_terms=info.so_terms,
                            p_hgvs=info.p_hgvs,
                            dosage=item.dosage,
                        ),
                        reference=parsed.reference_sequence or "",
                        position=start_position(parsed, change),
                        gene=gene,
                    )
                )
        nodes.sort(key=lambda n: (n.reference, n.position or 0))
        return nodes

    def build(
        self,
        gene: object,
        genotype: object,
        alleles: object = None,
        annotation_id: object = None,
    ) -> Data:
        """The genotype graph for one row (always at least one gene node)."""
        genes = split_values(gene)
        linked, functions = self.link(genes, genotype, alleles, annotation_id)
        variants = self._variant_nodes(linked, genes)

        self.stats["rows"] += 1
        self.stats["rows_with_variants"] += bool(variants)
        for item in linked:
            self.stats[f"variants_from_{item.source}"] += 1

        n_genes = max(len(genes), 1)
        x = [gene_node_vector() for _ in range(n_genes)]
        tokens = [self.gene_vocab.get(g, UNK_TOKEN) for g in genes] or [UNK_TOKEN]
        src: list[int] = []
        dst: list[int] = []
        attr: list[list[float]] = []

        def connect(a: int, b: int, features: list[float]) -> None:
            src.extend((a, b))
            dst.extend((b, a))
            attr.extend((features, features))

        previous: _Node | None = None
        for i, node in enumerate(variants, start=n_genes):
            x.append(node.features)
            tokens.append(PAD_TOKEN)
            hubs = [genes.index(node.gene)] if node.gene else range(n_genes)
            for hub in hubs:
                connect(hub, i, [1.0, 0.0, 0.0])
            if (
                previous is not None
                and previous.reference == node.reference
                and previous.position is not None
                and node.position is not None
            ):
                distance = abs(node.position - previous.position)
                connect(
                    i - 1,
                    i,
                    [
                        0.0,
                        1.0,
                        min(math.log10(1 + distance) / _MAX_DISTANCE_LOG10, 1.0),
                    ],
                )
            previous = node

        data = Data(
            x=torch.tensor(x, dtype=torch.float32),
            edge_index=torch.tensor([src, dst], dtype=torch.long).view(2, -1),
            edge_attr=torch.tensor(attr, dtype=torch.float32).view(-1, GENO_EDGE_DIM),
        )
        data.gene_token = torch.tensor(tokens, dtype=torch.long)
        data.geno_function = torch.tensor(
            [_mean(functions) if functions else [0.0] * PATH_FUNCTION_DIM],
            dtype=torch.float32,
        )
        data.hgvs = [item.hgvs for item in linked]
        return data

    def log_coverage(self) -> None:
        """Log how many rows reached at least one HGVS variant, and via what."""
        rows = self.stats["rows"] or 1
        logger.info(
            "Genotype HGVS coverage: %d/%d rows with variants (%.1f%%); %s",
            self.stats["rows_with_variants"],
            self.stats["rows"],
            100 * self.stats["rows_with_variants"] / rows,
            {
                k: v
                for k, v in sorted(self.stats.items())
                if k.startswith(("var", "un"))
            },
        )


def _mean(vectors: list[list[float]]) -> list[float]:
    return [sum(col) / len(vectors) for col in zip(*vectors)]


__all__ = [
    "GENO_EDGE_DIM",
    "PAD_TOKEN",
    "UNK_TOKEN",
    "AlleleCall",
    "GenotypeGraphBuilder",
    "build_gene_vocab",
    "gene_vocab_size",
    "split_values",
]
