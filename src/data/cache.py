"""Drug graph cache + empty-graph generator for ``DoubleTowerDataset``.

A ``GraphCache`` serves drug ``Data`` objects by drug ID (ChEMBL parent ID).
Drugs live in the Modephere library (``Settings.modephere.library``), keyed by
InChIKey: the cache maps the ID to its InChIKey, loads the compound through
``modephere.featurize.molecule_data`` (2D atom graph: ``x``, ``edge_index``,
``edge_attr``, ``global_feats``, ``z``) and attaches the ADMET-AI profile as
``admet_feats``. The genotype side does not go through this cache — genotypes
are HGVS variant graphs (``src.data.genotype``); the geno *placeholder*
(``make_empty_graph('geno', …)``) is kept for tests and tools.

Why split:
    - Easier to swap caching strategy (LRU, on-GPU, sharded by worker, …).
    - Easier to test cache-hit / cache-miss / corrupt-file paths in isolation.
    - The empty-graph helper now has dimensions baked in via a typed ``GraphDims``.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Literal

import torch
from modephere.types.layouts import MOLECULE_GRAPH_V1
from torch_geometric.data.data import Data

from src.data.library.admet import DRUG_ADMET_DIM, AdmetProvider

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from modephere.library import LibraryStore

    from src.config import Settings

logger = logging.getLogger(__name__)


# ``drug_id`` is the tag attr the dataset preserves on dummy graphs so
# downstream collators don't blow up; ``variant_name`` is the gene-graph
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

    Drug defaults are Modephere's molecule-graph layout
    (``modephere.types.layouts.MOLECULE_GRAPH_V1``) and the ADMET profile
    width (``src.data.library.admet.DRUG_ADMET_DIM``); genotype defaults come
    from ``src/data/genotype/`` (``HGVS_NODE_DIM`` / ``GENO_EDGE_DIM``).
    ``models.toml`` may override any of them via :meth:`from_extras`. The
    widths size the dummy-graph filler (so a missing artifact never breaks
    batching) and the model's input/aux branches.
    """

    drug_features: int = MOLECULE_GRAPH_V1.node_dim
    drug_edges: int = MOLECULE_GRAPH_V1.edge_dim
    drug_global: int = MOLECULE_GRAPH_V1.global_dim  # per-molecule descriptors
    drug_admet: int = DRUG_ADMET_DIM  # predicted ADMET/CYP profile
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
    data.name = name
    if kind == "drug":
        data.drug_id = str(graph_id)
        # Match the graph-level vectors real drug graphs carry, so a missing drug
        # never breaks batching of the ``global_feats`` / ``admet_feats`` attrs.
        data.global_feats = torch.zeros((1, d.drug_global), dtype=torch.float)
        data.admet_feats = torch.zeros((1, d.drug_admet), dtype=torch.float)
    if kind == "geno":
        data.graph_id = str(graph_id)
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


#: Identity attributes of a drug graph: kept at inference, dropped for training
#: (strings confuse PyG batching).
DRUG_IDENTITY_ATTRS = ("drug_id", "inchikey")


class GraphCache:
    """In-RAM cache over the Modephere drug library.

    ``drug_index`` maps a drug ID (ChEMBL parent ID) to its library InChIKey;
    ``store`` is the Modephere ``LibraryStore`` the graphs load from and
    ``admet`` the ADMET-AI profiles. A drug outside the index, or one that
    fails to load, gets an empty placeholder of the right widths.
    ``inference_mode`` keeps the identity attributes (``drug_id``,
    ``inchikey``) on returned graphs; training drops them.
    """

    def __init__(
        self,
        drug_index: Mapping[str, str],
        *,
        store: LibraryStore | None = None,
        admet: AdmetProvider | None = None,
        dims: GraphDims | None = None,
        inference_mode: bool = False,
    ) -> None:
        self.drug_index = dict(drug_index)
        self.store = store
        self.admet = admet or AdmetProvider.null()
        self.dims = dims or GraphDims()
        self.inference_mode = inference_mode

        self._drug_cache: dict[str, Data] = {}
        self._stats = {"drug_hits": 0, "drug_misses": 0}

    @classmethod
    def from_settings(
        cls,
        settings: Settings | None = None,
        *,
        dims: GraphDims | None = None,
        inference_mode: bool = False,
    ) -> GraphCache:
        """The cache over ``Settings.modephere.library`` and the library's ADMET table."""
        from src.config import get_settings
        from src.data.library.bridge import chembl_index, library_store
        from src.data.library.config import ADMET_FILE

        s = settings or get_settings()
        store = library_store(s.modephere)
        index = chembl_index(store)
        logger.info("Indexed %d drugs in %s", len(index), store.root)
        return cls(
            index,
            store=store,
            admet=AdmetProvider.load(s.paths.library / ADMET_FILE),
            dims=dims,
            inference_mode=inference_mode,
        )

    # ----- lookup ---------------------------------------------------------- #

    def get_drug(self, drug_id: str) -> Data:
        """The drug graph of ``drug_id``, or an empty placeholder."""
        drug_id = str(drug_id)
        data = self._drug_cache.get(drug_id)
        if data is None:
            self._stats["drug_misses"] += 1
            data = self._load(drug_id)
            if data is None:
                return make_empty_graph("drug", graph_id=drug_id, dims=self.dims)
            self._drug_cache[drug_id] = data
        else:
            self._stats["drug_hits"] += 1
        if not self.inference_mode:
            return data
        out = data.clone()
        out.drug_id = drug_id
        out.inchikey = self.drug_index[drug_id]
        return out

    def _load(self, drug_id: str) -> Data | None:
        """Molecule graph + ADMET profile of a stored drug (identity stripped)."""
        from modephere.featurize import molecule_data

        inchikey = self.drug_index.get(drug_id)
        if inchikey is None or self.store is None:
            return None
        try:
            data = molecule_data(self.store.load(inchikey))
        except Exception as e:  # noqa: BLE001 -- a bad file must not stop training
            logger.warning("Cannot load drug %s (%s): %s", drug_id, inchikey, e)
            return None
        data.admet_feats = self.admet.vector_for(inchikey)
        for attr in DRUG_IDENTITY_ATTRS:
            if attr in data:
                del data[attr]
        return _sanitize(data)

    # ----- preload --------------------------------------------------------- #

    def preload_drugs(self, drug_ids: Iterable[str]) -> None:
        for drug_id in drug_ids:
            drug_id = str(drug_id)
            if drug_id in self._drug_cache:
                continue
            data = self._load(drug_id)
            if data is not None:
                self._drug_cache[drug_id] = data

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
