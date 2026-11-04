"""Top-level orchestrator for a library build.

Composes the drug graph builder and the HGVS catalog builder and owns the
manifest lifecycle. Use as either:

    >>> LibraryBuilder(LibraryBuildConfig.from_settings()).run()

or via the CLI in ``src.data.library.__main__``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.data.genotype.catalog import build_hgvs_catalog
from src.data.library.admet import AdmetProvider, records_from_rows
from src.data.library.drugs import DrugGraphBuilder, load_drug_records
from src.data.library.manifest import BuildManifest
from src.data.library.pharmacophore import PharmacophoreGraphBuilder

if TYPE_CHECKING:
    from src.data.library.config import LibraryBuildConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BuildSummary:
    drugs_built: int
    drugs_skipped: int
    drugs_failed: int
    genes_built: int
    genes_failed: int
    pharm_built: int = 0
    pharm_skipped: int = 0
    pharm_failed: int = 0


class LibraryBuilder:
    """End-to-end orchestration of a library build.

    Single public method ``run()``; everything else is internal. The class
    instance is cheap to construct, so callers can build multiple libraries
    side-by-side (e.g. one per cohort) by reusing ``LibraryBuildConfig``s.
    """

    def __init__(self, config: LibraryBuildConfig) -> None:
        self.config = config

    def _build_admet_provider(self, cfg: LibraryBuildConfig) -> AdmetProvider:
        """Build (or load-from-cache) the ADMET profile provider for the drugs.

        With ``skip_admet`` the provider is null (every drug gets a zero
        ``admet_feats``); otherwise the catalog SMILES are run through ADMET-AI
        once and cached at ``cfg.admet_cache``.
        """
        if cfg.skip_admet:
            logger.info("ADMET prediction skipped (skip_admet=True) — zero profiles.")
            return AdmetProvider.null()

        records = records_from_rows(load_drug_records(cfg.drugs_tsv))
        logger.info(
            "ADMET: preparing profile for %d catalog drugs (cache=%s, force=%s)",
            len(records),
            cfg.admet_cache,
            cfg.force_admet,
        )
        return AdmetProvider.from_records(
            records,
            cfg.admet_cache,
            strip_salts=cfg.strip_salts,
            force=cfg.force_admet,
        )

    def run(self) -> BuildSummary:
        cfg = self.config
        cfg.ensure_outputs()

        manifest = BuildManifest.load_or_empty(cfg.manifest_path)
        logger.info(
            "Library build starting → %s (force=%s)",
            cfg.library_root,
            cfg.force,
        )

        drugs_built = drugs_skipped = drugs_failed = 0
        if not cfg.skip_drugs:
            admet = self._build_admet_provider(cfg)
            drug_builder = DrugGraphBuilder(
                cfg.drugs_out,
                force=cfg.force,
                failures_log=cfg.log_failures_path,
                saturation_log=cfg.log_saturation_path,
                strip_salts=cfg.strip_salts,
                admet=admet,
            )
            drugs_built, drugs_skipped, drugs_failed = drug_builder.build(
                cfg.drugs_tsv, manifest=manifest
            )
            if admet.misses:
                logger.warning(
                    "ADMET: %d drug graphs got a zero profile (CID absent from the "
                    "ADMET table).",
                    admet.misses,
                )
            manifest.save(cfg.manifest_path)
        else:
            logger.info("Skipping drug pipeline (skip_drugs=True).")

        pharm_built = pharm_skipped = pharm_failed = 0
        if not cfg.skip_pharm:
            pharm_builder = PharmacophoreGraphBuilder(
                cfg.pharm_out,
                force=cfg.force,
                failures_log=cfg.log_pharm_failures_path,
                strip_salts=cfg.strip_salts,
            )
            pharm_built, pharm_skipped, pharm_failed = pharm_builder.build(
                cfg.drugs_tsv, manifest=manifest
            )
            manifest.save(cfg.manifest_path)
        else:
            logger.info("Skipping pharmacophore pipeline (skip_pharm=True).")

        genes_built = genes_failed = 0
        if not cfg.skip_genes:
            try:
                catalog = build_hgvs_catalog(
                    snp_table=cfg.variants_tsv,
                    pharmvar_dir=cfg.pgx_dir,
                    kg_dir=cfg.kg_dir,
                    star_alleles_tsv=cfg.star_alleles_tsv,
                )
                catalog.save(cfg.hgvs_catalog_path)
                genes_built = len(catalog.genes())
            except Exception:
                logger.exception("HGVS catalog build failed")
                genes_failed = 1
            manifest.stats.genes_built = genes_built
            manifest.stats.genes_failed = genes_failed
        else:
            logger.info("Skipping HGVS catalog (skip_genes=True).")

        manifest.save(cfg.manifest_path)
        logger.info(
            "Library build done → %d drugs (+%d skipped, -%d failed), "
            "HGVS catalog over %d genes (failed=%d). Manifest at %s",
            drugs_built,
            drugs_skipped,
            drugs_failed,
            genes_built,
            genes_failed,
            cfg.manifest_path,
        )
        return BuildSummary(
            drugs_built=drugs_built,
            drugs_skipped=drugs_skipped,
            drugs_failed=drugs_failed,
            genes_built=genes_built,
            genes_failed=genes_failed,
            pharm_built=pharm_built,
            pharm_skipped=pharm_skipped,
            pharm_failed=pharm_failed,
        )
