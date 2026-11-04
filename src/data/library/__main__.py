"""CLI for the library builder.

Usage::

    python -m src.data.library
    python -m src.data.library --skip-drugs        # HGVS catalog only
    python -m src.data.library --skip-drugs --force
    python -m src.data.library --variants-tsv data/custom_snps.tsv

Drugs become one graph file each under ``drugs/``; the genotype side is the
``hgvs_catalog.pt`` that links rsIDs / star alleles / PharmGKB annotations to
HGVS (run ``python -m src.kg export`` first so the KG links are included).

Defaults are derived from the project ``Settings`` so most invocations are
zero-argument.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.core import setup_logging
from src.data.library.builder import LibraryBuilder
from src.data.library.config import LibraryBuildConfig

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.data.library",
        description="Build the offline drug + variant graph library.",
    )
    p.add_argument(
        "--variants-tsv",
        type=Path,
        default=None,
        help="Pan-gene SNP table (default: data/snp_data_output.tsv).",
    )
    p.add_argument(
        "--drugs-tsv",
        type=Path,
        default=None,
        help="Path to the drug catalog: .tsv/.csv or .json "
        "(default: data/drugs_cid.tsv).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing .pt files instead of skipping them.",
    )
    p.add_argument("--skip-drugs", action="store_true", help="Skip the drug pipeline.")
    p.add_argument("--skip-genes", action="store_true", help="Skip the HGVS catalog.")
    p.add_argument(
        "--skip-pharm", action="store_true", help="Skip the pharmacophore graphs."
    )
    p.add_argument(
        "--skip-admet",
        action="store_true",
        help="Skip ADMET prediction; drug graphs get a zero admet_feats vector "
        "(no GPU / fast builds).",
    )
    p.add_argument(
        "--force-admet",
        action="store_true",
        help="Recompute the ADMET cache even if a valid one already exists.",
    )
    p.add_argument(
        "--keep-salts",
        action="store_true",
        help="Keep salt counterions (do not reduce SMILES to the largest fragment).",
    )
    p.add_argument("--verbose", "-v", action="store_true", help="DEBUG-level logging.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    setup_logging(name="library", level=logging.DEBUG if args.verbose else logging.INFO)

    cfg = LibraryBuildConfig.from_settings(
        variants_tsv=args.variants_tsv,
        drugs_tsv=args.drugs_tsv,
        force=args.force,
        skip_drugs=args.skip_drugs,
        skip_genes=args.skip_genes,
        skip_admet=args.skip_admet,
        force_admet=args.force_admet,
        strip_salts=not args.keep_salts,
        skip_pharm=args.skip_pharm,
    )

    summary = LibraryBuilder(cfg).run()
    logger.info(
        "Drugs: built=%d skipped=%d failed=%d | Pharm: built=%d skipped=%d failed=%d "
        "| Genes: built=%d failed=%d",
        summary.drugs_built,
        summary.drugs_skipped,
        summary.drugs_failed,
        summary.pharm_built,
        summary.pharm_skipped,
        summary.pharm_failed,
        summary.genes_built,
        summary.genes_failed,
    )
    ok = summary.drugs_failed == 0 and summary.pharm_failed == 0
    return 0 if ok and summary.genes_failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
