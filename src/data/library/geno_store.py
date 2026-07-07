"""Single-file genotype-graph store + path/diplotype subgraph extraction.

The whole genotype library — every per-gene variation graph — is held in one
``.pt`` file (``torch.save`` of ``{gene: Data}``). A genotype is encoded by
materialising the **subgraph** induced by its path(s): the reference anchor plus
the union of the selected star alleles' variant nodes, re-chained in genomic
order. One label = a haplotype; two labels = a diplotype.

A custom dict store is used rather than ``InMemoryDataset.collate`` because each
gene graph carries non-tensor metadata (``paths``, ``path_function``,
``node_hgvs``) that PyG collation cannot fold into ``data.pt``/``slices``; the
single-file, indexed-by-gene goal is met all the same.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch_geometric.data.data import Data

from src.data.library.gene_graph import chain_edges
from src.data.library.haplotype_function import PATH_FUNCTION_DIM

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    from src.data.library.genotype_resolver import GenotypeResolver

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1


def _mean_vectors(vectors: Sequence[list[float]]) -> list[float]:
    """Element-wise mean of equal-length vectors (empty → a single zero vector)."""
    if not vectors:
        return [0.0] * PATH_FUNCTION_DIM
    dim = len(vectors[0])
    n = len(vectors)
    return [sum(v[i] for v in vectors) / n for i in range(dim)]


class GenoLibrary:
    """In-memory genotype library: ``gene -> per-gene variation graph``."""

    def __init__(
        self,
        graphs: dict[str, Data],
        rsid_to_hgvs: dict[str, str] | None = None,
    ) -> None:
        self._graphs = graphs
        self.rsid_to_hgvs = rsid_to_hgvs or {}

    def __contains__(self, gene: str) -> bool:
        return gene in self._graphs

    def __len__(self) -> int:
        return len(self._graphs)

    @property
    def genes(self) -> list[str]:
        return list(self._graphs)

    def get(self, gene: str) -> Data | None:
        return self._graphs.get(gene)

    def labels(self, gene: str) -> list[str]:
        """Star-allele/path labels available for ``gene``."""
        graph = self._graphs.get(gene)
        return list(graph.paths) if graph is not None else []

    # ----- persistence ----------------------------------------------------- #

    def resolver(self) -> GenotypeResolver:
        """Build a :class:`GenotypeResolver` over this library + its rsID bridge."""
        from src.data.library.genotype_resolver import GenotypeResolver

        return GenotypeResolver(self, self.rsid_to_hgvs)

    # ----- persistence ----------------------------------------------------- #

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "version": _SCHEMA_VERSION,
                "graphs": self._graphs,
                "rsid_to_hgvs": self.rsid_to_hgvs,
            },
            path,
        )
        logger.info(
            "GenoLibrary: saved %d gene graphs (+%d rsID bridges) to %s",
            len(self._graphs),
            len(self.rsid_to_hgvs),
            path,
        )

    @classmethod
    def load(cls, path: Path) -> GenoLibrary:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(blob, dict) and "graphs" in blob:
            graphs = blob["graphs"]
            rsid_to_hgvs = blob.get("rsid_to_hgvs", {})
        else:  # legacy: bare {gene: Data}
            graphs = blob
            rsid_to_hgvs = {}
        logger.info("GenoLibrary: loaded %d gene graphs from %s", len(graphs), path)
        return cls(graphs, rsid_to_hgvs)

    # ----- encoding -------------------------------------------------------- #

    def _require(self, gene: str) -> Data:
        base = self._graphs.get(gene)
        if base is None:
            msg = f"gene {gene!r} not in GenoLibrary"
            raise KeyError(msg)
        return base

    def _subgraph(
        self, base: Data, selected: list[int], geno_function: list[float]
    ) -> Data:
        """Induced subgraph over ``selected`` node indices, re-chained in order."""
        positions = [base.node_pos[i] for i in selected]
        edge_index, edge_attr = chain_edges(positions, int(base.gene_length))
        data = Data(
            x=base.x[selected].clone(), edge_index=edge_index, edge_attr=edge_attr
        )
        data.geno_function = torch.tensor([geno_function], dtype=torch.float32)
        return data

    def encode(self, gene: str, labels: str | Iterable[str]) -> Data:
        """Materialise the subgraph for a haplotype (one label) or diplotype (two).

        The subgraph is the anchor plus the union of the labels' variant nodes,
        re-chained in genomic order. The graph-level ``geno_function`` ([1, 6]) is
        the mean of the selected alleles' PGx-function vectors. Raises ``KeyError``
        when the gene is absent from the library.
        """
        base = self._require(gene)
        label_list = [labels] if isinstance(labels, str) else list(labels)
        selected = sorted(
            {0}.union(*(set(base.paths.get(label, ())) for label in label_list))
            if label_list
            else {0}
        )
        functions = [
            base.path_function[label]
            for label in label_list
            if label in base.path_function
        ]
        data = self._subgraph(base, selected, _mean_vectors(functions))
        data.gene = gene
        data.labels = label_list
        return data

    def encode_variants(self, gene: str, g_hgvs: Iterable[str]) -> Data:
        """Materialise the subgraph for an ad-hoc set of genomic-HGVS variants.

        Used for genotypes given by rsID (resolved to HGVS upstream): the path is
        the anchor plus whichever of the requested variants exist as nodes in the
        gene graph. Carries no star-allele label, so ``geno_function`` is zero.
        """
        base = self._require(gene)
        index = {hgvs: i for i, hgvs in enumerate(base.node_hgvs)}
        selected = sorted({0} | {index[h] for h in g_hgvs if h in index})
        data = self._subgraph(base, selected, [0.0] * PATH_FUNCTION_DIM)
        data.gene = gene
        data.labels = []
        return data


__all__ = ["GenoLibrary"]
