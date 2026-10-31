"""Graph cache + empty-graph generator extracted from ``DoubleTowerDataset``.

A ``GraphCache`` owns the in-memory copies of drug ``Data`` objects and the
on-disk lookup logic. The dataset asks the cache for ``get_drug(id)``. The
genotype side does not go through this cache — genotypes are built as HGVS
variant graphs by ``src.data.genotype`` — so the cache is drug-only; the geno
*placeholder* (``make_empty_graph('geno', …)``) is kept for tests and tools.

Why split:
    - Easier to swap caching strategy (LRU, on-GPU, sharded by worker, …).
    - Easier to test cache-hit / cache-miss / corrupt-file paths in isolation.
    - The empty-graph helper now has dimensions baked in via a typed ``GraphDims``.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Literal

import torch
from torch_geometric.data.data import Data

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path

logger = logging.getLogger(__name__)


# ``cid`` and ``smiles`` are tag attrs the dataset preserves on dummy graphs
# so downstream collators don't blow up; ``variant_name`` is the gene-graph
# equivalent.
_DRUG_DUMMY_NAME = "dummy_drug"
_GENO_DUMMY_NAME = "dummy_variant"


#: ``models.toml`` extras key -> ``GraphDims`` field.
_EXTRAS_KEYS: dict[str, str] = {
    "drug_node_features": "drug_features",
    "drug_attrs_features": "drug_edges",
    "drug_global_features": "drug_global",
    "drug_admet_features": "drug_admet",
    "geno_node_features": "geno_features",
    "geno_attrs_features": "geno_edges",
    "geno_function_features": "geno_function",
}


@dataclass(frozen=True)
class GraphDims:
    """Feature widths of the drug + genotype graphs — the single source of truth.

    Defaults match the graph library builders (``DRUG_NODE_DIM`` /
    ``DRUG_EDGE_DIM`` in ``src/data/library/drugs.py``, ``HGVS_NODE_DIM`` /
    ``GENO_EDGE_DIM`` in ``src/data/genotype/``). ``models.toml``
    may override any of them via :meth:`from_extras`. The widths size the
    dummy-graph filler (so a missing artifact never breaks batching) and the
    model's input/aux branches.
    """

    drug_features: int = 61
    drug_edges: int = 18
    drug_global: int = 1038  # per-molecule descriptor vector (drugs only)
    drug_admet: int = 41  # predicted ADMET/CYP profile (drugs only)
    geno_features: int = 45  # HGVS_NODE_DIM (src/data/genotype/features.py)
    geno_edges: int = 3  # GENO_EDGE_DIM: [membership, adjacency, distance]
    geno_function: int = 6  # star-allele PGx function vector (PATH_FUNCTION_DIM)

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, int) or value < 0:
                msg = f"invalid graph dimension {name}={value!r} (must be int >= 0)"
                raise ValueError(msg)

    @classmethod
    def from_extras(cls, extras: Mapping[str, object]) -> GraphDims:
        """Build from a model's ``models.toml`` extras, defaulting absent keys."""
        return cls(
            **{
                field: extras[key]  # type: ignore[misc]
                for key, field in _EXTRAS_KEYS.items()
                if key in extras
            }
        )


def make_empty_graph(
    kind: Literal["drug", "geno"],
    graph_id: str = "",
    *,
    dims: GraphDims | None = None,
) -> Data:
    """Build a 1-node, 0-edge placeholder graph with the configured dim.

    The placeholder has the right tensor shapes for the configured tower so
    a missing artifact never breaks PyG batching. Identifying metadata is
    attached so the predictor can still surface the failed key downstream.
    """
    d = dims or GraphDims()
    if kind == "drug":
        n_feat, n_edge = d.drug_features, d.drug_edges
        name = _DRUG_DUMMY_NAME
    elif kind == "geno":
        n_feat, n_edge = d.geno_features, d.geno_edges
        name = _GENO_DUMMY_NAME
    else:
        msg = f"unknown graph kind {kind!r}; must be 'drug' or 'geno'"
        raise ValueError(msg)

    data = Data(
        x=torch.zeros((1, n_feat), dtype=torch.float),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0, n_edge), dtype=torch.float),
    )
    data.cid = str(graph_id)
    data.smiles = ""
    data.name = name
    if kind == "drug":
        # Match the graph-level vectors real drug graphs carry, so a missing drug
        # never breaks batching of the ``global_feats`` / ``admet_feats`` attrs.
        data.global_feats = torch.zeros((1, d.drug_global), dtype=torch.float)
        data.admet_feats = torch.zeros((1, d.drug_admet), dtype=torch.float)
    if kind == "geno":
        # Match the attrs real genotype graphs carry (one UNK gene node), so a
        # placeholder never breaks batching of ``geno_function`` / ``gene_token``.
        data.geno_function = torch.zeros((1, d.geno_function), dtype=torch.float)
        data.gene_token = torch.ones(1, dtype=torch.long)
        data.gene = str(graph_id)
    return _sanitize(data)


def _sanitize(data: Data) -> Data:
    """Force tensors contiguous so multiprocessing DataLoaders stay happy."""
    if hasattr(data, "x") and data.x is not None:
        data.x = data.x.contiguous()
    if hasattr(data, "edge_index") and data.edge_index is not None:
        data.edge_index = data.edge_index.contiguous()
    if hasattr(data, "edge_attr") and data.edge_attr is not None:
        data.edge_attr = data.edge_attr.contiguous()
    return data


class PackedGraphs:
    """Many small graphs stored as a handful of contiguous tensors.

    A list of thousands of ``Data`` objects holds thousands of tensor storages;
    DataLoader workers receive each one as a separate shared-memory handle and
    exhaust the file-descriptor limit. Packing keeps a fixed number of
    storages; :meth:`__getitem__` rebuilds a graph from views.

    ``node_keys`` are per-node tensors (dim 0 = nodes), ``edge_attr_keys``
    per-edge (dim 0 = edges), ``graph_keys`` per-graph (dim 0 = 1).
    ``edge_index`` is stored with graph-local indices.
    """

    def __init__(
        self,
        graphs: list[Data],
        *,
        node_keys: tuple[str, ...] = ("x",),
        edge_attr_keys: tuple[str, ...] = ("edge_attr",),
        graph_keys: tuple[str, ...] = (),
    ) -> None:
        self.node_keys, self.edge_attr_keys, self.graph_keys = (
            node_keys,
            edge_attr_keys,
            graph_keys,
        )
        node_counts = [g.num_nodes for g in graphs]
        edge_counts = [g.edge_index.size(1) for g in graphs]
        self.node_ptr = _ptr(node_counts)
        self.edge_ptr = _ptr(edge_counts)
        self.edge_index = torch.cat([g.edge_index for g in graphs], dim=1)
        self.tensors = {
            key: torch.cat([g[key] for g in graphs], dim=0)
            for key in (*node_keys, *edge_attr_keys, *graph_keys)
        }

    def __len__(self) -> int:
        return len(self.node_ptr) - 1

    def __getitem__(self, i: int) -> Data:
        n0, n1 = int(self.node_ptr[i]), int(self.node_ptr[i + 1])
        e0, e1 = int(self.edge_ptr[i]), int(self.edge_ptr[i + 1])
        data = Data(edge_index=self.edge_index[:, e0:e1])
        for key in self.node_keys:
            data[key] = self.tensors[key][n0:n1]
        for key in self.edge_attr_keys:
            data[key] = self.tensors[key][e0:e1]
        for key in self.graph_keys:
            data[key] = self.tensors[key][i : i + 1]
        return data


def _ptr(counts: list[int]) -> torch.Tensor:
    ptr = torch.zeros(len(counts) + 1, dtype=torch.long)
    if counts:
        ptr[1:] = torch.tensor(counts, dtype=torch.long).cumsum(0)
    return ptr


_GC_INTERVAL = 1000  # how often to gc.collect() during a mass preload


class GraphCache:
    """In-RAM cache over the on-disk **drug** graph library.

    Holds the drug index plus a dict of loaded ``Data`` objects.
    ``inference_mode`` controls whether identifying metadata (cid, name,
    smiles) is preserved on returned graphs — training drops it (clutters
    batches), inference keeps it. The genotype tower is served by
    ``src.data.genotype.GenotypeGraphBuilder``, not by this cache.
    """

    def __init__(
        self,
        drug_index: dict[str, Path],
        *,
        dims: GraphDims | None = None,
        inference_mode: bool = False,
    ) -> None:
        self.drug_index = drug_index
        self.dims = dims or GraphDims()
        self.inference_mode = inference_mode

        self._drug_cache: dict[str, Data] = {}
        self._stats = {"drug_hits": 0, "drug_misses": 0}

    # ----- lookup ---------------------------------------------------------- #

    def get_drug(self, drug_id: str) -> Data:
        """Return the drug graph for ``drug_id`` or an empty placeholder."""
        return self._get(
            cache=self._drug_cache,
            key=drug_id,
            path=self.drug_index.get(drug_id),
            kind="drug",
        )

    def _get(
        self,
        cache: dict[str, Data],
        key: str,
        path: Path | None,
        kind: Literal["drug", "geno"],
    ) -> Data:
        # Cache hit
        if key in cache:
            self._stats[f"{kind}_hits"] += 1
            data = cache[key]
            return data.clone() if self.inference_mode else data

        self._stats[f"{kind}_misses"] += 1

        # Miss → try disk
        if path is not None and path.exists():
            try:
                data = torch.load(path, map_location="cpu", weights_only=False)
            except Exception as e:  # noqa: BLE001
                logger.warning("Corrupt graph file %s: %s", path, e)
                return make_empty_graph(kind, graph_id=key, dims=self.dims)

            if self.inference_mode:
                if not hasattr(data, "cid"):
                    data.cid = str(key)
            else:
                # Strip metadata in training mode — it confuses PyG batching.
                for attr in ("cid", "name", "smiles"):
                    if hasattr(data, attr):
                        delattr(data, attr)
            return _sanitize(data)

        # Miss + no path → placeholder
        return make_empty_graph(kind, graph_id=key, dims=self.dims)

    # ----- preload --------------------------------------------------------- #

    def preload_drugs(self, drug_ids: Iterable[str]) -> None:
        for i, drug_id in enumerate(drug_ids):
            path = self.drug_index.get(str(drug_id))
            if path is None:
                continue
            try:
                self._drug_cache[str(drug_id)] = torch.load(
                    path, map_location="cpu", weights_only=False
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to preload drug %s: %s", drug_id, e)
            if i and i % _GC_INTERVAL == 0:
                gc.collect()

    # ----- diagnostics ----------------------------------------------------- #

    @property
    def cached_drug_count(self) -> int:
        return len(self._drug_cache)

    def stats(self) -> dict[str, int | float]:
        """Return raw drug counters and hit rate."""
        total_drug = self._stats["drug_hits"] + self._stats["drug_misses"]
        return {
            **self._stats,
            "drug_hit_rate": self._stats["drug_hits"] / total_drug
            if total_drug
            else 0.0,
        }
