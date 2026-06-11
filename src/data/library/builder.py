"""Top-level orchestrator for a library build.

Composes the drug + gene builders, owns the manifest lifecycle, and applies
post-processing (file organization). Use as either:

    >>> LibraryBuilder(LibraryBuildConfig.from_settings()).run()

or via the CLI in ``src.data.library.__main__``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.data.library.admet import AdmetProvider, records_from_rows
from src.data.library.drugs import DrugGraphBuilder, load_drug_records
from src.data.library.genes import GenomicGraphBuilder
from src.data.library.geno_func import GenoFuncProvider
from src.data.library.manifest import BuildManifest

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

    def _build_geno_func_provider(self, cfg: LibraryBuildConfig) -> GenoFuncProvider:
        """Build the per-variant functional provider for the genotype tower.

        With ``skip_geno_func`` the provider is null (every variant gets a zero
        ``geno_global_feats``); otherwise Layer A (PGx allele function) is loaded
        from the local star-allele table and Layer B (pathogenicity) from the
        optional AlphaMissense / CADD files when configured.
        """
        if cfg.skip_geno_func:
            logger.info("Geno functional profile skipped — zero vectors.")
            return GenoFuncProvider.null()

        logger.info(
            "GenoFunc: building profile (star_alleles=%s, alphamissense=%s, cadd=%s)",
            cfg.star_alleles_tsv,
            cfg.alphamissense_path,
            cfg.cadd_path,
        )
        return GenoFuncProvider.from_sources(
            cfg.star_alleles_tsv,
            alphamissense_path=cfg.alphamissense_path,
            cadd_path=cfg.cadd_path,
        )

    def run(self) -> BuildSummary:
        cfg = self.config
        cfg.ensure_outputs()

        manifest = BuildManifest.load_or_empty(cfg.manifest_path)
        logger.info(
            "Library build starting → %s (force=%s, only_gene=%s)",
            cfg.library_root,
            cfg.force,
            cfg.only_gene,
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

        genes_built = genes_failed = 0
        if not cfg.skip_genes:
            func_provider = self._build_geno_func_provider(cfg)
            gene_builder = GenomicGraphBuilder(
                cfg.fasta_path,
                cfg.pgx_dir,
                only_gene=cfg.only_gene,
                force=cfg.force,
                func_provider=func_provider,
            )
            genes_built, genes_failed = gene_builder.build(
                cfg.variants_tsv,
                cfg.genes_out,
                manifest=manifest,
            )
            if func_provider.misses:
                logger.warning(
                    "GenoFunc: %d variant graphs got a zero functional profile "
                    "(no PGx-function and no pathogenicity annotation).",
                    func_provider.misses,
                )
            manifest.save(cfg.manifest_path)
        else:
            logger.info("Skipping gene pipeline (skip_genes=True).")

        manifest.save(cfg.manifest_path)
        logger.info(
            "Library build done → %d drugs (+%d skipped, -%d failed), "
            "%d gene variants (-%d failed). Manifest at %s",
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
        )
