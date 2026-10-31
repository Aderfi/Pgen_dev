"""Resolution report: what resolved, how, and — when not — why.

Writes ``resolution_report.tsv`` (one row per PharmGKB allele label) and
``disagreements.tsv`` (PharmVar vs dbSNP cross-check findings), and logs the
headline numbers: resolved label mentions, counts per kind x status x reason,
and the strand-mismatch rate of the allele-specific inference.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from pathlib import Path

    from src.kg.build import KGBuild

logger = logging.getLogger(__name__)

RESOLUTION_FILE = "resolution_report.tsv"
DISAGREEMENTS_FILE = "disagreements.tsv"


def write_report(build: KGBuild, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    resolution = pl.DataFrame(
        [r.model_dump() for r in build.resolution],
        schema={
            "label": pl.String,
            "kind": pl.String,
            "status": pl.String,
            "reason": pl.String,
            "n_annotations": pl.Int64,
            "n_variants": pl.Int64,
            "path": pl.String,
        },
    ).sort(["status", "n_annotations"], descending=[False, True])
    path = out_dir / RESOLUTION_FILE
    resolution.write_csv(path, separator="\t")

    disagreements = pl.DataFrame(
        [
            {**d.model_dump(), "dbsnp_spdis": ",".join(d.dbsnp_spdis)}
            for d in build.disagreements
        ],
        schema={
            "issue": pl.String,
            "haplotype": pl.String,
            "rsid": pl.String,
            "pharmvar_spdi": pl.String,
            "dbsnp_spdis": pl.String,
        },
    )
    disagreements.write_csv(out_dir / DISAGREEMENTS_FILE, separator="\t")

    mentions = Counter()
    for r in build.resolution:
        mentions[r.status] += r.n_annotations
    total = sum(mentions.values())
    logger.info(
        "KG run %s: resolved mentions %d/%d (%.1f%%), %d labels",
        build.run.id,
        mentions["resolved"],
        total,
        100 * mentions["resolved"] / total if total else 0.0,
        len(build.resolution),
    )
    by_kind = Counter((r.kind, r.status, r.reason or "-") for r in build.resolution)
    for (kind, status, reason), n in sorted(by_kind.items()):
        logger.info("  %-9s %-10s %-28s %6d labels", kind, status, reason, n)

    attempted = sum(build.inference_counts.values())
    mismatches = build.inference_counts.get("allele_strand_mismatch", 0)
    logger.info(
        "Allele inference: %d attempted, %d linked, strand mismatches %d/%d (%.4f)",
        attempted,
        build.inference_counts.get("linked", 0),
        mismatches,
        attempted,
        mismatches / attempted if attempted else 0.0,
    )
    for reason, n in sorted(build.inference_counts.items()):
        logger.info("  inference %-24s %6d", reason, n)
    for issue, n in sorted(Counter(d.issue for d in build.disagreements).items()):
        logger.info("PharmVar/dbSNP cross-check: %d %s", n, issue)
    logger.info("Reports written to %s", out_dir)
    return path


__all__ = ["DISAGREEMENTS_FILE", "RESOLUTION_FILE", "write_report"]
