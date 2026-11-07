"""Drug graph cache + empty-graph generator for ``DoubleTowerDataset``.

A ``GraphCache`` serves drug graphs (:class:`DrugData`) by drug ID (ChEMBL
parent ID). Drugs live in the Modephere library (``Settings.modephere.library``),
keyed by InChIKey: the cache maps the ID to its InChIKey, loads the compound
through ``modephere.featurize.molecule_data`` (2D atom graph: ``x``,
``edge_index``, ``edge_attr``, ``global_feats``, ``z``), attaches the ADMET-AI
profile as ``admet_feats`` and the pharmacophore graph
(``modephere.featurize.pharmacophore_data``) as ``pharm_*`` attributes, with
the feature-point positions of one conformer (``pharm_pos``; zeros and
``has_3d = 0`` for a 2D-only compound). Training samples a conformer per call;
:meth:`GraphCache.with_conformer` fixes one (inference averages over the
lowest-energy ones). The genotype side does not go through this cache — genotypes
are HGVS variant graphs (``src.data.genotype``); the geno *placeholder*
(``make_empty_graph('geno', …)``) is kept for tests and tools.

Why split:
    - Easier to swap caching strategy (LRU, on-GPU, sharded by worker, …).
    - Easier to test cache-hit / cache-miss / corrupt-file paths in isolation.
    - The empty-graph helper now has dimensions baked in via a typed ``GraphDims``.
"""

from __future__ import annotations

import copy
import logging
import random
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Literal

import torch
from modephere.types.layouts import MOLECULE_GRAPH_V1, PHARMACOPHORE_GRAPH_V1
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
    "pharm_node_features": "pharm_features",
    "pharm_attrs_features": "pharm_edges",
    "pharm_global_features": "pharm_global",
    "geno_node_features": "geno_features",
    "geno_attrs_features": "geno_edges",
    "geno_function_features": "geno_function",
}


@dataclass(frozen=True)
class GraphDims:
    """Feature widths of the drug + genotype graphs — the single source of truth.

    Drug defaults are Modephere's molecule-graph and pharmacophore-graph
    layouts (``modephere.types.layouts.MOLECULE_GRAPH_V1`` /
    ``PHARMACOPHORE_GRAPH_V1``) and the ADMET profile width
    (``src.data.library.admet.DRUG_ADMET_DIM``); genotype defaults come
    from ``src/data/genotype/`` (``HGVS_NODE_DIM`` / ``GENO_EDGE_DIM``).
    ``models.toml`` may override any of them via :meth:`from_extras`. The
    widths size the dummy-graph filler (so a missing artifact never breaks
    batching) and the model's input/aux branches.
    """

    drug_features: int = MOLECULE_GRAPH_V1.node_dim
    drug_edges: int = MOLECULE_GRAPH_V1.edge_dim
    drug_global: int = MOLECULE_GRAPH_V1.global_dim  # per-molecule descriptors
    drug_admet: int = DRUG_ADMET_DIM  # predicted ADMET/CYP profile
    pharm_features: int = PHARMACOPHORE_GRAPH_V1.node_dim  # pharmacophore points
    pharm_edges: int = PHARMACOPHORE_GRAPH_V1.edge_dim  # topological point pairs
    pharm_global: int = PHARMACOPHORE_GRAPH_V1.global_dim  # per-molecule pharm vector
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


#: Entity kinds of a row's drug set: one drug, or one ATC class (pooled members).
ENTITY_DRUG = 0
ENTITY_CLASS = 1

#: Pharmacophore attributes of a :class:`DrugData`.
PHARM_ATTRS = (
    "pharm_x",
    "pharm_edge_index",
    "pharm_edge_attr",
    "pharm_global",
    "pharm_pos",
    "has_3d",
)


class DrugData(Data):
    """One drug: the 2D atom graph plus its pharmacophore graph.

    Atom graph: ``x [N, 61]``, ``edge_index``, ``edge_attr [E, 18]``,
    ``global_feats [1, 1038]``, ``admet_feats [1, 41]``. Pharmacophore:
    ``pharm_x [K, 11]``, ``pharm_edge_index [2, P]`` (point indices),
    ``pharm_edge_attr [P, 13]``, ``pharm_global [1, 258]``, ``pharm_pos [K, 3]``
    (one conformer; zeros without 3D) and ``has_3d [1]``. Batching offsets
    ``pharm_edge_index`` by the point count; collate with
    ``follow_batch=["pharm_x"]`` for the point -> molecule index.
    """

    def __inc__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        if key == "pharm_edge_index":
            return self.pharm_x.size(0)
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        if key == "pharm_edge_index":
            return 1
        return super().__cat_dim__(key, value, *args, **kwargs)


def _empty_pharmacophore(data: Data, dims: GraphDims) -> None:
    """Attach a K = 0 pharmacophore without 3D (dummy drugs)."""
    data.pharm_x = torch.zeros((0, dims.pharm_features), dtype=torch.float)
    data.pharm_edge_index = torch.empty((2, 0), dtype=torch.long)
    data.pharm_edge_attr = torch.empty((0, dims.pharm_edges), dtype=torch.float)
    data.pharm_global = torch.zeros((1, dims.pharm_global), dtype=torch.float)
    data.pharm_pos = torch.zeros((0, 3), dtype=torch.float)
    data.has_3d = torch.zeros(1, dtype=torch.float)


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

    data = (DrugData if kind == "drug" else Data)(
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
        _empty_pharmacophore(data, d)
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


@dataclass(frozen=True)
class _Compound:
    """Cached tensors of one stored compound (shared by every view)."""

    data: DrugData  # 2D graph + pharmacophore, pharm_pos zeros, has_3d 0
    positions: torch.Tensor | None  # [C, K, 3] point positions per conformer


class GraphCache:
    """In-RAM cache over the Modephere drug library.

    ``drug_index`` maps a drug ID (ChEMBL parent ID) to its library InChIKey;
    ``store`` is the Modephere ``LibraryStore`` the graphs load from and
    ``admet`` the ADMET-AI profiles. A drug outside the index, or one that
    fails to load, gets an empty placeholder of the right widths.
    ``inference_mode`` keeps the identity attributes (``drug_id``,
    ``inchikey``) on returned graphs; training drops them.

    ``class_members`` maps ATC codes to ChEMBL parent IDs (class entities,
    :meth:`get_entity`). ``conformer`` ``None`` samples a conformer per call
    with ``rng``; an int fixes conformer ``min(conformer, n - 1)`` (``0`` is the
    lowest-energy one). :meth:`with_conformer` / :meth:`without_members` return
    views that share the loaded tensors.
    """

    def __init__(
        self,
        drug_index: Mapping[str, str],
        *,
        store: LibraryStore | None = None,
        admet: AdmetProvider | None = None,
        dims: GraphDims | None = None,
        inference_mode: bool = False,
        class_members: Mapping[str, frozenset[str]] | None = None,
        conformer: int | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.drug_index = dict(drug_index)
        self.store = store
        self.admet = admet or AdmetProvider.null()
        self.dims = dims or GraphDims()
        self.inference_mode = inference_mode
        self.class_members: Mapping[str, frozenset[str]] = dict(class_members or {})
        self.conformer = conformer
        self.excluded: frozenset[str] = frozenset()
        self._rng = rng or random.Random()

        self._compounds: dict[str, _Compound] = {}
        self._failed: set[str] = set()
        self._stats = {"drug_hits": 0, "drug_misses": 0}

    @classmethod
    def from_settings(
        cls,
        settings: Settings | None = None,
        *,
        dims: GraphDims | None = None,
        inference_mode: bool = False,
    ) -> GraphCache:
        """The cache over ``Settings.modephere.library``, its ADMET table and the
        ATC members of Modephere's export (classes stay empty without it)."""
        from src.config import get_settings
        from src.core import ConfigurationError
        from src.data.library.bridge import atc_members, chembl_index, library_store
        from src.data.library.config import ADMET_FILE

        s = settings or get_settings()
        store = library_store(s.modephere)
        index = chembl_index(store)
        try:
            members = atc_members(s.modephere)
        except ConfigurationError as e:
            logger.warning("No ATC members, drug classes stay empty: %s", e)
            members = {}
        logger.info(
            "Indexed %d drugs in %s (%d ATC codes)",
            len(index),
            store.root,
            len(members),
        )
        return cls(
            index,
            store=store,
            admet=AdmetProvider.load(s.paths.library / ADMET_FILE),
            dims=dims,
            inference_mode=inference_mode,
            class_members=members,
        )

    # ----- views ----------------------------------------------------------- #

    def with_conformer(self, index: int | None) -> GraphCache:
        """A view serving conformer ``index`` (``None``: sampled per call)."""
        view = copy.copy(self)
        view.conformer = index
        return view

    def without_members(self, chembl_ids: Iterable[str]) -> GraphCache:
        """A view whose class pools leave out ``chembl_ids`` (drug lookups unchanged)."""
        view = copy.copy(self)
        view.excluded = self.excluded | frozenset(chembl_ids)
        return view

    # ----- lookup ---------------------------------------------------------- #

    def entity_members(self, key: str) -> list[str]:
        """Drug IDs an entity key reads that are in the library index.

        ``CHEMBL…``: the drug itself; ``ATC:…``: the class members, minus the
        :meth:`without_members` exclusions, sorted.
        """
        from src.data.library.bridge import class_members, is_class_key

        if is_class_key(key):
            members = class_members(key, self.class_members) - self.excluded
            return sorted(m for m in members if m in self.drug_index)
        return [key] if key in self.drug_index else []

    def get_entity(self, key: str) -> tuple[int, list[DrugData]]:
        """``(kind, molecules)`` of an entity key; molecules that fail to load
        are left out (an unbuilt drug or an empty class gives ``[]``)."""
        from src.data.library.bridge import is_class_key

        kind = ENTITY_CLASS if is_class_key(key) else ENTITY_DRUG
        drugs = [
            self.get_drug(m)
            for m in self.entity_members(key)
            if self._compound(m, count=False) is not None
        ]
        return kind, drugs

    def n_conformers(self, drug_id: str) -> int:
        """Conformers stored for ``drug_id`` (0: 2D-only or not in the library)."""
        compound = self._compound(str(drug_id), count=False)
        if compound is None or compound.positions is None:
            return 0
        return int(compound.positions.size(0))

    def get_drug(self, drug_id: str) -> DrugData:
        """The drug graph of ``drug_id`` (one conformer), or an empty placeholder."""
        drug_id = str(drug_id)
        compound = self._compound(drug_id)
        if compound is None:
            return make_empty_graph("drug", graph_id=drug_id, dims=self.dims)  # type: ignore[return-value]
        # inference callers may modify what they get: a deep copy keeps the
        # cache intact; training shares the cached tensors
        data = (
            compound.data.clone() if self.inference_mode else copy.copy(compound.data)
        )
        if compound.positions is not None:
            n = compound.positions.size(0)
            i = (
                self._rng.randrange(n)
                if self.conformer is None
                else min(self.conformer, n - 1)
            )
            data.pharm_pos = compound.positions[i]
            data.has_3d = torch.ones(1, dtype=torch.float)
        if self.inference_mode:
            data.drug_id = drug_id
            data.inchikey = self.drug_index[drug_id]
        return data

    def _compound(self, drug_id: str, *, count: bool = True) -> _Compound | None:
        inchikey = self.drug_index.get(drug_id)
        if inchikey is None or inchikey in self._failed:
            if count:
                self._stats["drug_misses"] += 1
            return None
        compound = self._compounds.get(inchikey)
        if count:
            self._stats["drug_hits" if compound is not None else "drug_misses"] += 1
        if compound is None:
            compound = self._load(drug_id, inchikey)
            if compound is None:
                self._failed.add(inchikey)
                return None
            self._compounds[inchikey] = compound
        return compound

    def _load(self, drug_id: str, inchikey: str) -> _Compound | None:
        """Molecule + pharmacophore graph, ADMET and conformer positions."""
        from modephere.featurize import (
            feature_positions,
            molecule_data,
            pharmacophore_data,
        )

        if self.store is None:
            return None
        try:
            art = self.store.load(inchikey)
            mol = molecule_data(art)
            pharm = pharmacophore_data(art)
            n_conf = 0 if art.conformers is None else len(art.pharmacophores_3d)
            positions = (
                torch.stack([feature_positions(art, c) for c in range(n_conf)])
                if n_conf
                else None
            )
        except Exception as e:  # noqa: BLE001 -- a bad file must not stop training
            logger.warning("Cannot load drug %s (%s): %s", drug_id, inchikey, e)
            return None
        data = DrugData(
            x=mol.x,
            edge_index=mol.edge_index,
            edge_attr=mol.edge_attr,
            global_feats=mol.global_feats,
            z=mol.z,
        )
        data.admet_feats = self.admet.vector_for(inchikey)
        data.pharm_x = pharm.x.float()
        data.pharm_edge_index = pharm.edge_index.long()
        data.pharm_edge_attr = pharm.edge_attr.float()
        data.pharm_global = pharm.global_feats.float().reshape(1, -1)
        data.pharm_pos = torch.zeros((pharm.x.size(0), 3), dtype=torch.float)
        data.has_3d = torch.zeros(1, dtype=torch.float)
        return _Compound(data=_sanitize(data), positions=positions)

    # ----- preload --------------------------------------------------------- #

    def preload_drugs(self, drug_ids: Iterable[str]) -> None:
        for drug_id in drug_ids:
            self._compound(str(drug_id), count=False)

    # ----- diagnostics ----------------------------------------------------- #

    @property
    def cached_drug_count(self) -> int:
        return len(self._compounds)

    def stats(self) -> dict[str, int | float]:
        """Return raw drug counters and hit rate."""
        total_drug = self._stats["drug_hits"] + self._stats["drug_misses"]
        return {
            **self._stats,
            "drug_hit_rate": self._stats["drug_hits"] / total_drug
            if total_drug
            else 0.0,
        }
