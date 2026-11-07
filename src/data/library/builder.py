"""One library build: Modephere drugs, ADMET profiles and the HGVS catalog.

    >>> LibraryBuilder(LibraryBuildConfig.from_settings()).run()

Steps, each skippable:

1. ``refresh_pgx`` — re-export Modephere's pgx tables from its Neo4j;
2. drugs — the ChEMBL IDs of every annotation drug go through Modephere's
   ``LibraryBuilder`` (offline, against the staged ChEMBL release); compounds
   already stored are kept unless ``force``;
3. ADMET — profiles for every stored compound the ADMET table lacks;
4. genes — the HGVS catalog from the pgx export + the star-allele table.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from modephere.library import LibraryBuilder as ModephereLibraryBuilder

from src.data.genotype.catalog import build_hgvs_catalog
from src.data.library.admet import update_admet_table
from src.data.library.bridge import (
    annotation_chembl_ids,
    library_build_config,
    library_store,
    refresh_pgx_exports,
)

if TYPE_CHECKING:
    from modephere.library import BuildSummary as DrugSummary

    from src.data.library.config import LibraryBuildConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BuildSummary:
    pgx_tables: dict[str, int] = field(default_factory=dict)
    drugs: DrugSummary | None = None
    admet_profiles: int = 0
    genes: int = 0
    catalog_failed: bool = False

    @property
    def ok(self) -> bool:
        """Every step that ran produced its output (drug failures are expected)."""
        built_any = self.drugs is None or self.drugs.built + self.drugs.skipped > 0
        return built_any and not self.catalog_failed


class LibraryBuilder:
    """Runs the steps of :mod:`src.data.library.builder` for one config."""

    def __init__(self, config: LibraryBuildConfig) -> None:
        self.config = config

    def _drugs(self) -> DrugSummary:
        cfg = self.config
        ids = annotation_chembl_ids(cfg.modephere)
        logger.info("Drug library: %d annotation ChEMBL IDs", len(ids))
        build = library_build_config(
            chembl_ids=ids, settings=cfg.modephere, force=cfg.force
        )
        return ModephereLibraryBuilder(build).run()

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
        drugs = None if cfg.skip_drugs else self._drugs()
        admet = 0 if cfg.skip_admet else self._admet()
        genes, failed = (0, False) if cfg.skip_genes else self._genes()
        summary = BuildSummary(
            pgx_tables=pgx,
            drugs=drugs,
            admet_profiles=admet,
            genes=genes,
            catalog_failed=failed,
        )
        if drugs is not None:
            logger.info(
                "Drugs: %d queries, %d built, %d already stored, %d failed (%s)",
                drugs.queries,
                drugs.built,
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


__all__ = ["BuildSummary", "LibraryBuilder"]
