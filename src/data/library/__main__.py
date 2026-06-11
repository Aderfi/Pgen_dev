"""CLI for the library builder.

Usage::

    python -m src.data.library
    python -m src.data.library --only-gene CYP2D6
    python -m src.data.library --skip-drugs --force
    python -m src.data.library --variants-tsv data/custom_snps.tsv

Defaults are derived from the project ``Settings`` so most invocations are
zero-argument.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.config import configure_logging_level, get_settings
from src.data.library.builder import LibraryBuilder
from src.data.library.config import LibraryBuildConfig


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.data.library",
        description="Build the offline drug + variant graph library.",
    )
    p.add_argument(
        "--variants-tsv",
        type=Path,
        default=None,
        help="Path to the variants TSV (default: data/snp_data_output.tsv).",
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
    p.add_argument(
        "--only-gene",
        type=str,
        default=None,
        metavar="SYMBOL",
        help="Build only the variants for this gene — useful for verification.",
    )
    p.add_argument("--skip-drugs", action="store_true", help="Skip the drug pipeline.")
    p.add_argument("--skip-genes", action="store_true", help="Skip the gene pipeline.")
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

    settings = get_settings()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        configure_logging_level(settings)
    logging.basicConfig(format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    cfg = LibraryBuildConfig.from_settings(
        variants_tsv=args.variants_tsv,
        drugs_tsv=args.drugs_tsv,
        force=args.force,
        only_gene=args.only_gene,
        skip_drugs=args.skip_drugs,
        skip_genes=args.skip_genes,
        skip_admet=args.skip_admet,
        force_admet=args.force_admet,
        strip_salts=not args.keep_salts,
    )

    summary = LibraryBuilder(cfg).run()
    print(
        f"Drugs:  built={summary.drugs_built} skipped={summary.drugs_skipped} "
        f"failed={summary.drugs_failed}\n"
        f"Genes:  built={summary.genes_built} failed={summary.genes_failed}"
    )
    return 0 if summary.drugs_failed == 0 and summary.genes_failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
