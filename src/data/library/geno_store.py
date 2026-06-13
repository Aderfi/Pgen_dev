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

    def __init__(self, graphs: dict[str, Data]) -> None:
        self._graphs = graphs

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

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"version": _SCHEMA_VERSION, "graphs": self._graphs}, path)
        logger.info("GenoLibrary: saved %d gene graphs to %s", len(self._graphs), path)

    @classmethod
    def load(cls, path: Path) -> GenoLibrary:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        graphs = blob["graphs"] if isinstance(blob, dict) and "graphs" in blob else blob
        logger.info("GenoLibrary: loaded %d gene graphs from %s", len(graphs), path)
        return cls(graphs)

    # ----- encoding -------------------------------------------------------- #

    def encode(self, gene: str, labels: str | Iterable[str]) -> Data:
        """Materialise the subgraph for a haplotype (one label) or diplotype (two).

        The subgraph is the anchor plus the union of the labels' variant nodes,
        re-chained in genomic order. The graph-level ``geno_function`` ([1, 6]) is
        the mean of the selected alleles' PGx-function vectors. Raises ``KeyError``
        when the gene is absent from the library.
        """
        base = self._graphs.get(gene)
        if base is None:
            msg = f"gene {gene!r} not in GenoLibrary"
            raise KeyError(msg)

        label_list = [labels] if isinstance(labels, str) else list(labels)
        selected = sorted(
            {0}.union(*(set(base.paths.get(label, ())) for label in label_list))
            if label_list
            else {0}
        )

        positions = [base.node_pos[i] for i in selected]
        edge_index, edge_attr = chain_edges(positions, int(base.gene_length))
        functions = [
            base.path_function[label]
            for label in label_list
            if label in base.path_function
        ]

        data = Data(
            x=base.x[selected].clone(),
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        data.geno_function = torch.tensor(
            [_mean_vectors(functions)], dtype=torch.float32
        )
        data.gene = gene
        data.labels = label_list
        return data


__all__ = ["GenoLibrary"]
