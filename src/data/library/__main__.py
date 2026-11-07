"""CLI for the library builder.

Usage::

    python -m src.data.library                  # drugs + ADMET + HGVS catalog
    python -m src.data.library --refresh-pgx    # re-export Modephere's pgx tables first
    python -m src.data.library --skip-drugs     # ADMET + catalog only
    python -m src.data.library --skip-drugs --skip-admet   # catalog only

Drugs are built by Modephere into ``Settings.modephere.library``
(``compounds/<InChIKey>.safetensors``); the HGVS catalog
(``paths.library/hgvs_catalog.pt``) is assembled from Modephere's pgx export
(``Settings.modephere.pgx_export``). Every location and the Neo4j connection
come from ``src/config/data/settings.toml`` ``[modephere]`` /
``PHARMAGEN_MODEPHERE__*``.
"""

from __future__ import annotations

import argparse
import logging
import sys

from src.core import setup_logging
from src.data.library.builder import LibraryBuilder
from src.data.library.config import LibraryBuildConfig

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.data.library",
        description="Build the drug library (Modephere), ADMET profiles and the "
        "HGVS catalog.",
    )
    p.add_argument(
        "--refresh-pgx",
        action="store_true",
        help="Re-export Modephere's pgx tables from its Neo4j (modephere.kg) first.",
    )
    p.add_argument(
        "--force", action="store_true", help="Rebuild drugs that are already stored."
    )
    p.add_argument("--skip-drugs", action="store_true", help="Skip the drug build.")
    p.add_argument("--skip-genes", action="store_true", help="Skip the HGVS catalog.")
    p.add_argument(
        "--skip-admet",
        action="store_true",
        help="Leave the ADMET table as it is (no GPU / fast builds).",
    )
    p.add_argument(
        "--force-admet", action="store_true", help="Recompute every ADMET profile."
    )
    p.add_argument("--verbose", "-v", action="store_true", help="DEBUG-level logging.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    setup_logging(name="library", level=logging.DEBUG if args.verbose else logging.INFO)
    cfg = LibraryBuildConfig.from_settings(
        refresh_pgx=args.refresh_pgx,
        force=args.force,
        skip_drugs=args.skip_drugs,
        skip_genes=args.skip_genes,
        skip_admet=args.skip_admet,
        force_admet=args.force_admet,
    )
    summary = LibraryBuilder(cfg).run()
    return 0 if summary.ok else 2


if __name__ == "__main__":
    sys.exit(main())
