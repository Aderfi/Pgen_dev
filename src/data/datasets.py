"""PyTorch Datasets for Pharmagen.

:class:`DoubleTowerDataset` — the active two-tower GNN dataset. Composes
``GraphCache`` (in-RAM graph store) and ``TargetEncoder`` (target
tensorization).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl
import torch
from torch.utils.data import Dataset

from src.config import get_settings
from src.core import DataError
from src.data.cache import GraphCache, GraphDims, make_empty_graph
from src.data.collator import PolyData
from src.data.encoders import TargetEncoder
from src.data.graph_indexing import GraphIndexBuilder
from src.data.library.geno_store import GenoLibrary

if TYPE_CHECKING:
    from collections.abc import Iterable, Set

    from src.data.library.genotype_resolver import GenotypeResolver
    from src.data.polypharmacy import PseudoPatientBuilder

logger = logging.getLogger(__name__)

_LIBRARY = get_settings().paths.library
_GENO_LIBRARY_FILE = "geno_graphs.pt"

# Threshold above which RAM preloading is suspicious (warn, don't refuse).
PRELOAD_THRESHOLD = 5000

# Default tower dimensions when callers don't override.
DEFAULT_DIMENSIONS: dict[str, dict[str, int]] = {
    "drugs": {"features": 61, "edges": 18, "attrs": 0, "global": 1038, "admet": 41},
    "geno": {"features": 30, "edges": 2, "attrs": 0, "function": 6},
}


# --------------------------------------------------------------------------- #
# DoubleTowerDataset — active GNN dataset.
# --------------------------------------------------------------------------- #


def _dims_from_input(input_dimensions: dict[str, dict[str, int]] | None) -> GraphDims:
    """Translate the legacy nested-dict dim spec into a typed GraphDims."""
    if not input_dimensions:
        return GraphDims()

    drugs = input_dimensions.get("drugs", {})
    geno = input_dimensions.get("geno", {})
    return GraphDims(
        drug_features=drugs.get("features", DEFAULT_DIMENSIONS["drugs"]["features"]),
        drug_edges=drugs.get("edges", DEFAULT_DIMENSIONS["drugs"]["edges"]),
        drug_global=drugs.get("global", DEFAULT_DIMENSIONS["drugs"]["global"]),
        drug_admet=drugs.get("admet", DEFAULT_DIMENSIONS["drugs"]["admet"]),
        geno_features=geno.get("features", DEFAULT_DIMENSIONS["geno"]["features"]),
        geno_edges=geno.get("edges", DEFAULT_DIMENSIONS["geno"]["edges"]),
        geno_function=geno.get("function", DEFAULT_DIMENSIONS["geno"]["function"]),
    )


def _validate_input_dims(dims: dict[str, dict[str, int]]) -> None:
    required_subkeys = ("features", "edges", "attrs")
    for kind in ("drugs", "geno"):
        if kind not in dims:
            continue
        if not isinstance(dims[kind], dict):
            msg = f"dimension {kind!r} must be a dict, got {type(dims[kind]).__name__}"
            raise DataError(msg)
        for subkey in required_subkeys:
            if subkey not in dims[kind]:
                continue
            v = dims[kind][subkey]
            if not isinstance(v, int) or v < 0:
                msg = (
                    f"invalid dimension {kind}.{subkey}: {v} (must be non-negative int)"
                )
                raise DataError(msg)


class DoubleTowerDataset(Dataset):
    """Drug-graph + genotype-graph pair dataset for the Two-Tower GNN.

    Composition:
        - ``GraphIndexBuilder``  — discovers the on-disk drug library.
        - ``GraphCache``         — owns the in-memory drug cache + dummy fallback.
        - ``GenotypeResolver``   — resolves ``(gene, genotype)`` to a subgraph.
        - ``TargetEncoder``      — fits/applies sklearn encoders to target cols.

    The dataset itself just glues these together and exposes ``__getitem__``.

    Args:
        df: Polars DataFrame with the join columns.
        drug_col: Column name containing drug IDs.
        geno_col: Column name containing the genotype string (star/rsID tokens).
        target_cols: Target column names.
        multilabel_cols: Subset of ``target_cols`` that are multi-label.
        encoders: Pre-fitted encoders (REQUIRED for val/test sets to keep
            the same class layout as training).
        gene_col: Column name containing the HGNC gene symbol.
        genotype_resolver: Resolver over the ``GenoLibrary`` (loaded from
            config when omitted; shared across train/val for one RAM copy).
        drug_lib: Override drug library path (default from config).
        preload_ram: If True, eagerly loads all unique drug graphs into RAM.
        input_dimensions: Legacy nested-dict dim spec; converts to ``GraphDims``.
        type_data: Unused, kept for back-compat.
        inference_mode: If True, preserves identifying metadata on returned graphs.
        pseudo_patient_builder: Optional ``PseudoPatientBuilder``. When given,
            ``__getitem__`` returns a molecule-level polypharmacy drug graph
            (focal drug + kept DDI neighbours as a :class:`~src.data.collator.PolyData`)
            instead of the single-molecule drug graph. When omitted (the
            default), behaviour is unchanged — single-molecule path only.

            NOTE: this emits one row per molecule using each molecule's
            precomputed ``global_feats`` descriptor, not a full atom-level
            encoding of every molecule. Wiring the true two-level
            atom -> molecule -> patient batching (running the atom-level
            drug-tower GNN per molecule, then this poly graph over the
            resulting molecule embeddings) is deferred to a later phase.
    """

    def __init__(
        self,
        df: pl.DataFrame,
        drug_col: str,
        geno_col: str,
        target_cols: list[str],
        multilabel_cols: Iterable[str] | Set[str],
        encoders: dict[str, Any] | None = None,
        gene_col: str = "gene",
        genotype_resolver: GenotypeResolver | None = None,
        drug_lib: Path = _LIBRARY / "drugs",
        preload_ram: bool = False,
        input_dimensions: dict[str, dict[str, int]] | None = None,
        type_data: str | None = None,  # noqa: ARG002 (legacy arg)
        inference_mode: bool = False,
        pseudo_patient_builder: PseudoPatientBuilder | None = None,
    ) -> None:
        # 1. Frame
        if isinstance(df, pl.LazyFrame):
            logger.info("Collecting LazyFrame for Dataset access ...")
            self.df = df.collect()
        elif isinstance(df, pl.DataFrame):
            self.df = df
        else:
            msg = f"df must be a Polars DataFrame, got {type(df).__name__}"
            raise TypeError(msg)

        if input_dimensions:
            _validate_input_dims(input_dimensions)
        self.dims = _dims_from_input(input_dimensions)

        if preload_ram and len(self.df) > PRELOAD_THRESHOLD:
            logger.warning(
                "preload_ram=True with %d samples may cause OOM (threshold: %d).",
                len(self.df),
                PRELOAD_THRESHOLD,
            )

        self.drug_col = drug_col
        self.geno_col = geno_col
        self.gene_col = gene_col
        self.target_cols = target_cols
        self.multilabel_cols = set(multilabel_cols) if multilabel_cols else set()
        self.inference_mode = inference_mode
        self.pseudo_patient_builder = pseudo_patient_builder

        # 2. Drug index + cache
        drug_index = GraphIndexBuilder.build_drug_index(drug_lib)
        logger.info("Indexed %d drugs", len(drug_index))
        self.cache = GraphCache(
            drug_index=drug_index,
            dims=self.dims,
            inference_mode=inference_mode,
        )

        # 2b. Genotype resolver over the single-file GenoLibrary.
        self.resolver = genotype_resolver or self._default_resolver()

        # 3. Encoders + targets
        self.target_encoder = TargetEncoder(
            target_cols=target_cols,
            multilabel_cols=self.multilabel_cols,
            encoders=dict(encoders) if encoders else None,
        )
        self.targets = self.target_encoder.fit_transform(self.df)

        # 4. Random-access lookups (avoid Polars indexing per __getitem__)
        self.lookup_drugs = self.df[self.drug_col].to_list()
        self.lookup_genes = (
            self.df[self.gene_col].to_list()
            if self.gene_col in self.df.columns
            else [""] * len(self.df)
        )
        self.lookup_genotypes = self.df[self.geno_col].to_list()

        # 5. Optional drug preload
        if preload_ram:
            unique_drugs = (
                self.df.select(pl.col(self.drug_col).unique().cast(pl.String))
                .to_series()
                .to_list()
            )
            logger.info("Preloading %d drugs into RAM ...", len(unique_drugs))
            self.cache.preload_drugs(unique_drugs)
            logger.info("Cached %d drugs", self.cache.cached_drug_count)

    @staticmethod
    def _default_resolver() -> GenotypeResolver:
        """Load the project ``GenoLibrary`` and build a resolver over it."""
        return GenoLibrary.load(_LIBRARY / _GENO_LIBRARY_FILE).resolver()

    # ----- Dataset API ----------------------------------------------------- #

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        drug_cid = str(self.lookup_drugs[idx])
        if self.pseudo_patient_builder is not None:
            drug = self._build_poly_drug_data(drug_cid)
        else:
            drug = self.cache.get_drug(drug_cid)
        gene = str(self.lookup_genes[idx])
        genotype = str(self.lookup_genotypes[idx])
        geno = self.resolver.resolve(gene, genotype)
        if geno is None:
            geno = make_empty_graph("geno", graph_id=gene, dims=self.dims)
        # String attrs (gene/labels) are stripped by DoubleTowerCollater before
        # batching; in inference mode it extracts them as ids first.
        targets = {col: self.targets[col][idx] for col in self.target_cols}
        return {"drug_data": drug, "geno_data": geno, "targets": targets}

    def _build_poly_drug_data(self, focal_cid: str) -> PolyData:
        """Build a molecule-level polypharmacy drug graph for one patient.

        Delegates neighbour selection to ``self.pseudo_patient_builder`` and
        packs the result into a :class:`PolyData` so
        :class:`~src.data.collator.DoubleTowerCollater` offsets
        ``ddi_edge_index``/``mol_to_patient`` correctly across a batch.

        Deferred (see class docstring): each molecule contributes one row of
        its precomputed ``global_feats`` descriptor, not an atom-level graph
        run through the drug tower — full two-level atom -> molecule ->
        patient batching is left for a later phase.

        WARNING: the emitted ``PolyData`` is NOT yet consumable by
        ``PharmagenTwoTower.forward`` under a realistic config: the descriptor
        lives in ``x`` (not in a ``global_feats`` attribute), and ``x`` has the
        molecule-descriptor width, not ``drug_in_features``. Wiring this through
        the model requires the deferred two-level batching (or a
        molecule-descriptor-only forward path). Until then this builds a
        correctly *collatable* structure (the DDI-offset contract) but not a
        model-ready batch. The polypharmacy switches stay OFF by default.
        """
        builder = self.pseudo_patient_builder
        assert builder is not None  # narrows for type checkers; guarded by caller
        sample = builder.build(focal_cid)
        molecules = sample["molecules"]
        mol_x = torch.stack(
            [
                m.global_feats.reshape(-1)
                if getattr(m, "global_feats", None) is not None
                else torch.zeros(self.dims.drug_global)
                for m in molecules
            ]
        )
        drug = PolyData(x=mol_x, edge_index=torch.empty((2, 0), dtype=torch.long))
        drug.mol_to_patient = sample["mol_to_patient"]
        drug.ddi_edge_index = sample["ddi_edge_index"]
        drug.ddi_edge_attr = sample["ddi_edge_attr"]
        drug.is_focal = sample["is_focal"]
        if self.inference_mode:
            drug.cid = focal_cid
        return drug

    def get_cache_stats(self) -> dict[str, int | float]:
        """Cache hit/miss counters and rates (for logging / dashboards)."""
        return self.cache.stats()


__all__ = [
    "DEFAULT_DIMENSIONS",
    "DoubleTowerDataset",
    "PRELOAD_THRESHOLD",
]
