"""PyTorch Datasets for Pharmagen.

:class:`DoubleTowerDataset` — the active two-tower GNN dataset. Composes
``GraphCache`` (in-RAM graph store) and ``TargetEncoder`` (target
tensorization).
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, Any

import polars as pl
import torch
from torch.utils.data import Dataset

from src.data.cache import ENTITY_DRUG, GraphCache, GraphDims, PackedGraphs
from src.data.collator import PolyData
from src.data.encoders import TargetEncoder
from src.data.genotype import GenotypeGraphBuilder, load_hgvs_catalog

if TYPE_CHECKING:
    from collections.abc import Iterable, Set

    from src.data.polypharmacy import PseudoPatientBuilder

logger = logging.getLogger(__name__)

# Threshold above which RAM preloading is suspicious (warn, don't refuse).
PRELOAD_THRESHOLD = 5000


ALLELES_COLUMN = "alleles"
ANNOTATION_ID_COLUMN = "variant_annotation_id"
#: Entity keys of one row are ``|``-joined in the drug column.
ENTITY_SEP = "|"
#: Class members read per item in training (randomly sampled above this).
MAX_CLASS_MEMBERS = 64


class DoubleTowerDataset(Dataset):
    """Drug-graph + genotype-graph pair dataset for the Two-Tower GNN.

    Composition:
        - ``GraphCache``         — drug graphs from the Modephere library (by
          ChEMBL ID), their ADMET profiles, and the dummy fallback.
        - ``GenotypeGraphBuilder`` — reads each genotype as an HGVS variant graph.
        - ``TargetEncoder``      — fits/applies sklearn encoders to target cols.

    The dataset itself just glues these together and exposes ``__getitem__``.

    Args:
        df: Polars DataFrame with the join columns.
        drug_col: Column with the row's drug entity keys, ``|``-joined
            (``CHEMBL25``, ``ATC:N02A``). Each item's ``drug_data`` is a list
            of ``(kind, molecules)`` per readable entity (see
            :meth:`GraphCache.get_entity`); a row with none gets one dummy
            drug, counted in :attr:`rows_without_drug`.
        geno_col: Column with the genotype tokens (HGVS, rsIDs, star alleles).
        target_cols: Target column names.
        multilabel_cols: Subset of ``target_cols`` that are multi-label.
        encoders: Pre-fitted encoders (REQUIRED for val/test sets to keep
            the same class layout as training).
        gene_col: Column name containing the HGNC gene symbol.
        genotype_builder: HGVS genotype-graph builder (shared across splits so
            they use the same gene vocabulary). Default: the project catalog
            with an empty vocabulary (every gene → UNK).
        drug_cache: Drug graph cache to read from (share one across splits);
            default: :meth:`GraphCache.from_settings` (``Settings.modephere``).
        preload_ram: If True, eagerly loads all unique drug graphs into RAM.
        dims: Graph feature widths (sizes the placeholder graphs).
        inference_mode: If True, preserves identifying metadata on returned graphs.
        max_class_members: Class members per item in training; a bigger class
            is randomly sampled to this many (all members at inference).
        seed: Seed of the member sampling.
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
        genotype_builder: GenotypeGraphBuilder | None = None,
        drug_cache: GraphCache | None = None,
        preload_ram: bool = False,
        dims: GraphDims | None = None,
        inference_mode: bool = False,
        pseudo_patient_builder: PseudoPatientBuilder | None = None,
        max_class_members: int = MAX_CLASS_MEMBERS,
        seed: int | None = None,
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

        self.dims = dims or GraphDims()

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
        self.max_class_members = max_class_members
        self._rng = random.Random(seed)

        # 2. Drug cache (Modephere library + ADMET profiles)
        self.cache = drug_cache or GraphCache.from_settings(
            dims=self.dims, inference_mode=inference_mode
        )

        # 2b. Genotype graphs: built once per row (small, reused every epoch).
        builder = genotype_builder or GenotypeGraphBuilder(load_hgvs_catalog(), {})
        builder.stats.clear()
        graphs = [
            builder.build(*row) for row in zip(*self._genotype_columns(), strict=True)
        ]
        self.geno_hgvs = [g.hgvs for g in graphs]
        self.geno_graphs = PackedGraphs(
            graphs,
            node_keys=("x", "gene_token"),
            graph_keys=("geno_function",),
        )
        builder.log_coverage()

        # 3. Encoders + targets
        self.target_encoder = TargetEncoder(
            target_cols=target_cols,
            multilabel_cols=self.multilabel_cols,
            encoders=dict(encoders) if encoders else None,
        )
        self.targets = self.target_encoder.fit_transform(self.df)

        # 4. Random-access lookups (avoid Polars indexing per __getitem__)
        self.lookup_drugs = self.df[self.drug_col].to_list()
        self.lookup_entities = [
            [k for k in str(v or "").split(ENTITY_SEP) if k] for v in self.lookup_drugs
        ]
        self.rows_without_drug = sum(
            not any(self.cache.entity_members(k) for k in keys)
            for keys in self.lookup_entities
        )
        if self.rows_without_drug:
            logger.info(
                "%d rows have no readable drug (dummy graph)", self.rows_without_drug
            )

        # 5. Optional drug preload
        if preload_ram:
            unique_drugs = sorted(
                {
                    m
                    for keys in self.lookup_entities
                    for k in keys
                    for m in self.cache.entity_members(k)
                }
            )
            logger.info("Preloading %d drugs into RAM ...", len(unique_drugs))
            self.cache.preload_drugs(unique_drugs)
            logger.info("Cached %d drugs", self.cache.cached_drug_count)

    def _genotype_columns(self) -> tuple[list[Any], ...]:
        """(gene, genotype, alleles, annotation id) per row; absent columns → None."""
        n = len(self.df)

        def column(name: str) -> list[Any]:
            return self.df[name].to_list() if name in self.df.columns else [None] * n

        return (
            column(self.gene_col),
            column(self.geno_col),
            column(ALLELES_COLUMN),
            column(ANNOTATION_ID_COLUMN),
        )

    # ----- Dataset API ----------------------------------------------------- #

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self.pseudo_patient_builder is not None:
            drug: Any = self._build_poly_drug_data(str(self.lookup_drugs[idx]))
        else:
            drug = self._entities(idx)
        geno = self.geno_graphs[idx]
        targets = {col: self.targets[col][idx] for col in self.target_cols}
        return {"drug_data": drug, "geno_data": geno, "targets": targets}

    def _entities(self, idx: int) -> list[tuple[int, list[Any]]]:
        """``(kind, molecules)`` per readable entity of row ``idx``."""
        entities = []
        for key in self.lookup_entities[idx]:
            kind, molecules = self.cache.get_entity(key)
            if not molecules:
                continue
            if not self.inference_mode and len(molecules) > self.max_class_members:
                molecules = self._rng.sample(molecules, self.max_class_members)
            entities.append((kind, molecules))
        if not entities:
            dummy_id = self.lookup_entities[idx][0] if self.lookup_entities[idx] else ""
            entities = [(ENTITY_DRUG, [self.cache.get_drug(dummy_id)])]
        return entities

    def _build_poly_drug_data(self, focal_id: str) -> PolyData:
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
        sample = builder.build(focal_id)
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
            drug.drug_id = focal_id
        return drug

    def get_cache_stats(self) -> dict[str, int | float]:
        """Cache hit/miss counters and rates (for logging / dashboards)."""
        return self.cache.stats()


__all__ = [
    "ALLELES_COLUMN",
    "ANNOTATION_ID_COLUMN",
    "ENTITY_SEP",
    "MAX_CLASS_MEMBERS",
    "PRELOAD_THRESHOLD",
    "DoubleTowerDataset",
]
