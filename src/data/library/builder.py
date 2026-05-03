"""Top-level orchestrator for a library build.

Composes the drug + gene builders, owns the manifest lifecycle, and applies
post-processing (file organization). Use as either:

    >>> LibraryBuilder(LibraryBuildConfig.from_settings()).run()

or via the CLI in ``src.data.library.__main__``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.data.library.config import LibraryBuildConfig
from src.data.library.drugs import DrugGraphBuilder
from src.data.library.genes import GenomicGraphBuilder
from src.data.library.manifest import BuildManifest

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

    def run(self) -> BuildSummary:
        cfg = self.config
        cfg.ensure_outputs()

        manifest = BuildManifest.load_or_empty(cfg.manifest_path)
        logger.info(
            "Library build starting → %s (force=%s, only_gene=%s)",
            cfg.library_root, cfg.force, cfg.only_gene,
        )

        drugs_built = drugs_skipped = drugs_failed = 0
        if not cfg.skip_drugs:
            drug_builder = DrugGraphBuilder(
                cfg.drugs_out,
                force=cfg.force,
                failures_log=cfg.log_failures_path
                or (cfg.library_root / "build_failures.log"),
            )
            drugs_built, drugs_skipped, drugs_failed = drug_builder.build(
                cfg.drugs_tsv, manifest=manifest
            )
            manifest.save(cfg.manifest_path)
        else:
            logger.info("Skipping drug pipeline (skip_drugs=True).")

        genes_built = genes_failed = 0
        if not cfg.skip_genes:
            gene_builder = GenomicGraphBuilder(
                cfg.fasta_path,
                cfg.pgx_dir,
                only_gene=cfg.only_gene,
                force=cfg.force,
            )
            genes_built, genes_failed = gene_builder.build(
                cfg.variants_tsv, cfg.genes_out, manifest=manifest,
            )
            manifest.save(cfg.manifest_path)
        else:
            logger.info("Skipping gene pipeline (skip_genes=True).")

        manifest.save(cfg.manifest_path)
        logger.info(
            "Library build done → %d drugs (+%d skipped, -%d failed), "
            "%d gene variants (-%d failed). Manifest at %s",
            drugs_built, drugs_skipped, drugs_failed,
            genes_built, genes_failed, cfg.manifest_path,
        )
        return BuildSummary(
            drugs_built=drugs_built,
            drugs_skipped=drugs_skipped,
            drugs_failed=drugs_failed,
            genes_built=genes_built,
            genes_failed=genes_failed,
        )
