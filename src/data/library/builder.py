"""One library build: Modephere drugs, ADMET profiles and the HGVS catalog.

    >>> LibraryBuilder(LibraryBuildConfig.from_settings()).run()

Steps, each skippable:

1. ``refresh_pgx`` — re-export Modephere's pgx tables from its Neo4j;
2. drugs — the ChEMBL IDs the training rows read (single drugs, combination
   components, drug-class members) go through Modephere's ``LibraryBuilder``
   (offline, against the staged ChEMBL release); compounds already stored are
   kept unless ``force`` — except those built with another RDKit, which are
   always rebuilt (every compound the model reads comes from one toolkit);
3. ADMET — profiles for every stored compound the ADMET table lacks;
4. genes — the HGVS catalog from the pgx export + the star-allele table.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import polars as pl
import rdkit
from modephere.library import BuildSummary as DrugSummary
from modephere.library import LibraryBuilder as ModephereLibraryBuilder

from src.core import ConfigurationError
from src.data.genotype.catalog import build_hgvs_catalog
from src.data.library.admet import update_admet_table
from src.data.library.bridge import (
    annotation_chembl_ids,
    annotation_drugs,
    atc_members,
    chembl_index,
    class_entity_keys,
    class_members,
    library_build_config,
    library_store,
    refresh_pgx_exports,
)

if TYPE_CHECKING:
    from src.data.library.config import LibraryBuildConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BuildSummary:
    pgx_tables: dict[str, int] = field(default_factory=dict)
    drugs: DrugSummary | None = None
    admet_profiles: int = 0
    genes: int = 0
    catalog_failed: bool = False
    #: Stored compounds rebuilt because another RDKit version built them.
    stale_rebuilt: int = 0

    @property
    def ok(self) -> bool:
        """Every step that ran produced its output (drug failures are expected)."""
        built_any = self.drugs is None or self.drugs.built + self.drugs.skipped > 0
        return built_any and not self.catalog_failed


class LibraryBuilder:
    """Runs the steps of :mod:`src.data.library.builder` for one config."""

    def __init__(self, config: LibraryBuildConfig) -> None:
        self.config = config

    def _stale(self, ids: list[str]) -> set[str]:
        """IDs whose stored compound was built with another RDKit version."""
        store = library_store(self.config.modephere)
        index = chembl_index(store)
        compounds = store.read_table("compounds").select("inchikey", "toolkit_version")
        stale_keys = set(
            compounds.filter(pl.col("toolkit_version") != rdkit.__version__)[
                "inchikey"
            ].to_list()
        )
        return {i for i in ids if index.get(i) in stale_keys}

    def _drugs(self) -> tuple[DrugSummary, int]:
        """Build the catalog's drugs; returns the merged summary and the
        number of stale compounds rebuilt."""
        cfg = self.config
        ids = annotation_chembl_ids(cfg.modephere)
        stale = set() if cfg.force else self._stale(ids)
        rest = [i for i in ids if i not in stale]
        logger.info(
            "Drug library: %d ChEMBL IDs (%d stored with another RDKit than %s: "
            "rebuilt)",
            len(ids),
            len(stale),
            rdkit.__version__,
        )
        runs = []
        for chembl_ids, force in ((sorted(stale), True), (rest, cfg.force)):
            if not chembl_ids:
                continue
            build = library_build_config(
                chembl_ids=chembl_ids, settings=cfg.modephere, force=force
            )
            builder = ModephereLibraryBuilder(build)
            runs.append((builder, builder.run()))
        if len(runs) > 1:  # each run rewrites failures.parquet: keep both
            library_store(cfg.modephere).write_failures(
                [r for b, _ in runs for r in (*b.failures, *b.degraded)]
            )
        return _merge([s for _, s in runs]), len(stale)

    def _log_classes(self) -> None:
        """One line per drug class whose members are not all in the library."""
        cfg = self.config
        keys = class_entity_keys(annotation_drugs(cfg.modephere))
        if not keys:
            return
        try:
            members = atc_members(cfg.modephere)
        except ConfigurationError:
            return
        index = chembl_index(library_store(cfg.modephere))
        partial = 0
        for key in keys:
            pool = class_members(key, members)
            built = sum(m in index for m in pool)
            if built < len(pool):
                partial += 1
                logger.info("class %s: %d of %d members built", key, built, len(pool))
        logger.info(
            "Drug classes: %d, %d with members missing from the library",
            len(keys),
            partial,
        )

    def _admet(self) -> int:
        cfg = self.config
        compounds = library_store(cfg.modephere).read_table("compounds")
        records = compounds.select("inchikey", "working_smiles").iter_rows()
        return update_admet_table(records, cfg.admet_path, force=cfg.force_admet).height

    def _genes(self) -> tuple[int, bool]:
        cfg = self.config
        try:
            catalog = build_hgvs_catalog(
                pgx_export=cfg.modephere.pgx_export,
                star_alleles_tsv=cfg.star_alleles_tsv,
            )
            catalog.save(cfg.hgvs_catalog_path)
        except Exception:
            logger.exception("HGVS catalog build failed")
            return 0, True
        return len(catalog.genes()), False

    def run(self) -> BuildSummary:
        cfg = self.config
        logger.info(
            "Library build: drugs -> %s, catalog + ADMET -> %s",
            cfg.modephere.library,
            cfg.library_root,
        )
        pgx = refresh_pgx_exports(cfg.modephere) if cfg.refresh_pgx else {}
        drugs, stale = (None, 0) if cfg.skip_drugs else self._drugs()
        if drugs is not None:
            self._log_classes()
        admet = 0 if cfg.skip_admet else self._admet()
        genes, failed = (0, False) if cfg.skip_genes else self._genes()
        summary = BuildSummary(
            pgx_tables=pgx,
            drugs=drugs,
            admet_profiles=admet,
            genes=genes,
            catalog_failed=failed,
            stale_rebuilt=stale,
        )
        if drugs is not None:
            logger.info(
                "Drugs: %d queries, %d built (%d stale rebuilt, %d 2D-only), %d "
                "already stored, %d failed (%s)",
                drugs.queries,
                drugs.built,
                stale,
                drugs.degraded,
                drugs.skipped,
                drugs.failed,
                ", ".join(f"{k}={v}" for k, v in sorted(drugs.failures.items())) or "-",
            )
        logger.info(
            "Library build done: %d ADMET profiles, HGVS catalog over %d genes%s",
            admet,
            genes,
            " (FAILED)" if failed else "",
        )
        return summary


def _merge(summaries: list[DrugSummary]) -> DrugSummary:
    """One summary over several Modephere runs."""
    failures: Counter = Counter()
    for s in summaries:
        failures.update(s.failures)
    return DrugSummary(
        queries=sum(s.queries for s in summaries),
        built=sum(s.built for s in summaries),
        skipped=sum(s.skipped for s in summaries),
        duplicates=sum(s.duplicates for s in summaries),
        failed=sum(s.failed for s in summaries),
        failures=dict(failures),
        degraded=sum(s.degraded for s in summaries),
    )


__all__ = ["BuildSummary", "LibraryBuilder"]
