"""Graph cache + empty-graph generator extracted from ``DoubleTowerDataset``.

A ``GraphCache`` owns the in-memory copies of drug + variant ``Data`` objects
and the on-disk lookup logic. The dataset becomes a thin orchestrator that
asks the cache for ``get_drug(id)`` / ``get_variant(key)``.

Why split:
    - Easier to swap caching strategy (LRU, on-GPU, sharded by worker, …).
    - Easier to test cache-hit / cache-miss / corrupt-file paths in isolation.
    - The empty-graph helper now has dimensions baked in via a typed ``GraphDims``.
"""

from __future__ import annotations

import gc
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch_geometric.data.data import Data

logger = logging.getLogger(__name__)


# ``cid`` and ``smiles`` are tag attrs the dataset preserves on dummy graphs
# so downstream collators don't blow up; ``variant_name`` is the gene-graph
# equivalent.
_DRUG_DUMMY_NAME = "dummy_drug"
_GENO_DUMMY_NAME = "dummy_variant"


@dataclass(frozen=True)
class GraphDims:
    """Expected feature dimensions for the drug + genotype towers.

    Defaults match the trained TwoTowerGAT model. Override at construction
    if a future architecture uses different counts — the dimensions flow
    through into the dummy-graph filler so an empty drug/variant doesn't
    break batching.
    """

    drug_features: int = 61
    drug_edges: int = 18
    geno_features: int = 9
    geno_edges: int = 3


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
    if kind == "geno":
        data.variant_name = str(graph_id)
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


_GC_INTERVAL = 1000  # how often to gc.collect() during a mass preload


class GraphCache:
    """In-RAM cache over the on-disk graph library.

    Holds two indices (drug and variant) plus per-kind dicts of loaded
    ``Data`` objects. ``inference_mode`` controls whether identifying
    metadata (cid, name, smiles, variant_name) is preserved on returned
    graphs — training drops it (clutters batches), inference keeps it.
    """

    def __init__(
        self,
        drug_index: dict[str, Path],
        variant_index: dict[str, dict[str, Path]],
        *,
        dims: GraphDims | None = None,
        inference_mode: bool = False,
    ) -> None:
        self.drug_index = drug_index
        self.variant_index = variant_index
        self.dims = dims or GraphDims()
        self.inference_mode = inference_mode

        self._drug_cache: dict[str, Data] = {}
        self._geno_cache: dict[str, Data] = {}
        self._stats = {
            "drug_hits": 0,
            "drug_misses": 0,
            "geno_hits": 0,
            "geno_misses": 0,
        }

    # ----- lookup ---------------------------------------------------------- #

    def get_drug(self, drug_id: str) -> Data:
        """Return the drug graph for ``drug_id`` or an empty placeholder."""
        return self._get(
            cache=self._drug_cache,
            key=drug_id,
            path=self.drug_index.get(drug_id),
            kind="drug",
        )

    def get_variant(self, variant_key: str) -> Data:
        """Return the variant graph for a ``GENE_<variant>`` key."""
        path: Path | None = None
        if "_" in variant_key:
            gene, variant = variant_key.split("_", 1)
            path = self.variant_index.get(gene, {}).get(variant)
        return self._get(
            cache=self._geno_cache,
            key=variant_key,
            path=path,
            kind="geno",
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
                for attr in ("cid", "name", "smiles", "variant_name"):
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

    def preload_variants(self, variant_keys: Iterable[str]) -> None:
        for i, key in enumerate(variant_keys):
            if "_" not in key:
                continue
            gene, variant = key.split("_", 1)
            path = self.variant_index.get(gene, {}).get(variant)
            if path is None:
                continue
            try:
                self._geno_cache[key] = torch.load(
                    path, map_location="cpu", weights_only=False
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to preload variant %s: %s", key, e)
            if i and i % _GC_INTERVAL == 0:
                gc.collect()

    # ----- diagnostics ----------------------------------------------------- #

    @property
    def cached_drug_count(self) -> int:
        return len(self._drug_cache)

    @property
    def cached_variant_count(self) -> int:
        return len(self._geno_cache)

    def stats(self) -> dict[str, int | float]:
        """Return raw counters and hit rates for both kinds."""
        total_drug = self._stats["drug_hits"] + self._stats["drug_misses"]
        total_geno = self._stats["geno_hits"] + self._stats["geno_misses"]
        return {
            **self._stats,
            "drug_hit_rate": self._stats["drug_hits"] / total_drug
            if total_drug
            else 0.0,
            "geno_hit_rate": self._stats["geno_hits"] / total_geno
            if total_geno
            else 0.0,
        }
