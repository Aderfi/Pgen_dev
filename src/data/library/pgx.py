"""Loader for the PharmVar-style per-gene VCF folder.

Layout::

    data/haplotype_variants/
    ├── CYP2D6/
    │   ├── CYP2D6_4.vcf
    │   ├── CYP2D6_10.vcf
    │   └── ...
    ├── DPYD/
    │   ├── rs3918290.vcf       # the rsID-named DPYD variants are special-cased
    │   └── ...
    └── ...

Each VCF is treated as a haplotype definition: filename → haplotype label,
contents → variant rows. The rsID → star-allele mapping for DPYD is sourced
from ``data/dicts/star_alleles.tsv`` to avoid a hardcoded table.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import polars as pl

from src.genomics.star_alleles import get_default_map

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


_VCF_COLUMNS = ["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO"]


def parse_haplotype_label(gene: str, vcf_stem: str) -> str:
    """Translate a VCF filename stem into a haplotype label.

    Rules:
        - ``rsXXXX`` → look it up in the star-allele map; fall back to the rsID itself.
        - ``GENE_<n>`` / ``<n>``     → ``*<n>``  (digits become star alleles)
        - everything else            → returned unchanged after stripping the gene prefix.
    """
    star_map = get_default_map()

    if vcf_stem.startswith("rs"):
        alleles = star_map.alleles_for_rsid(vcf_stem)
        if alleles:
            # If multiple alleles share an rsID, prefer the one matching this gene.
            same_gene = [a for a in alleles if a.gene.symbol == gene]
            chosen = same_gene[0] if same_gene else alleles[0]
            return f"*{chosen.allele}"
        return vcf_stem

    # Strip "<gene>_" prefix or bare gene name.
    cleaned = vcf_stem
    if cleaned.startswith(f"{gene}_"):
        cleaned = cleaned[len(gene) + 1 :]
    elif cleaned == gene:
        cleaned = ""

    if not cleaned:
        return "*1"

    # ``c.2846A>T`` and similar HGVS strings are kept as-is.
    if cleaned[0].isdigit() or not cleaned.startswith("*"):
        return f"*{cleaned}"
    return cleaned


def load_pgx_folder(pgx_dir: Path) -> pl.DataFrame:
    """Concatenate all per-gene VCFs into a single Polars DataFrame.

    Returned columns: ``CHROM, POS, REF, ALT, gene_provided, haplotype_label``.
    Empty/missing folders return an empty DataFrame (not an error) so the
    caller can still proceed with TSV-only inputs.
    """
    if not pgx_dir.exists():
        logger.warning("PGx folder not found: %s — skipping VCF ingest.", pgx_dir)
        return pl.DataFrame()

    frames: list[pl.DataFrame] = []
    for gene_folder in sorted(pgx_dir.iterdir()):
        if not gene_folder.is_dir():
            continue
        gene = gene_folder.name
        vcf_files = sorted(gene_folder.glob("*.vcf"))
        if not vcf_files:
            continue

        for vcf_file in vcf_files:
            haplo_label = parse_haplotype_label(gene, vcf_file.stem)
            try:
                vcf_df = pl.read_csv(
                    vcf_file,
                    separator="\t",
                    comment_prefix="#",
                    has_header=False,
                    new_columns=_VCF_COLUMNS,
                    schema_overrides={"CHROM": pl.Utf8, "REF": pl.Utf8, "ALT": pl.Utf8},
                    ignore_errors=True,
                )
            except Exception as e:  # noqa: BLE001 — keep going on malformed files
                logger.warning("Could not parse VCF %s: %s — skipping.", vcf_file, e)
                continue

            if vcf_df.is_empty():
                continue

            frames.append(
                vcf_df.select(
                    [
                        "CHROM",
                        "POS",
                        "REF",
                        "ALT",
                        pl.lit(gene).alias("gene_provided"),
                        pl.lit(haplo_label).alias("haplotype_label"),
                    ]
                )
            )

    if not frames:
        return pl.DataFrame()
    combined = pl.concat(frames)
    logger.info(
        "PGx loader: %d variant rows across %d files.", len(combined), len(frames)
    )
    return combined
