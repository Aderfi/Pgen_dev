# Pharmagen - Pharmacogenetic Prediction and Therapeutic Efficacy
# Copyright (C) 2025 Adrim Hamed Outmani
# Licensed under the GNU GPLv3. See LICENSE file in the project root.
"""VCF parsing and variant validation against a reference FASTA.

Bioinformatics conventions:
    - VCF coordinates are 1-based; pysam exposes ``record.pos`` (1-based) and
      ``record.start`` (0-based, half-open). We compare REF using the
      half-open ``[start, stop)`` slice from pysam, which already matches
      the underlying FASTA bytes.
    - Multi-allelic sites and missing genotypes ('./.') are handled
      explicitly.
    - The interactive VCF picker is intentionally NOT in this module — it
      lives in the CLI layer (src/cli/workflows/genomics.py once Phase 4f
      lands). Library code never calls ``input()``.
"""

from __future__ import annotations

import logging
from collections.abc import Generator, Iterable
from pathlib import Path
from typing import Any

import pysam  # type: ignore[import-not-found]

from src.config import get_settings as _get_settings
from src.core import BioinformaticsError

_paths = _get_settings().paths
DATA_DIR = _paths.data
REF_GENOME_FASTA = _paths.ref_genome_fasta

logger = logging.getLogger(__name__)


# Default locations — keep available for legacy callers/scripts.
DEFAULT_FASTA: Path = REF_GENOME_FASTA
DEFAULT_VCF_DIR: Path = Path(DATA_DIR, "raw")


def list_vcf_files(vcf_dir: Path = DEFAULT_VCF_DIR) -> list[Path]:
    """Return the .vcf.gz files available under ``vcf_dir`` (sorted)."""
    if not vcf_dir.exists():
        return []
    return sorted(vcf_dir.glob("*.vcf.gz"))


def decode_genotype(record: Any, sample_id: str) -> dict[str, Any]:
    """Decode a single sample's genotype from a pysam VariantRecord.

    Handles missing calls, homozygous (ref/alt), heterozygous, and compound
    heterozygous (Alt1/Alt2) cases.

    Returns a dict with::

        {
            "type":   "Homozygous Reference" | "Homozygous Alternate"
                     | "Heterozygous" | "Compound Heterozygous"
                     | "Missing",
            "alleles": "A/G"  # or "./." for missing
        }
    """
    gt_tuple = record.samples[sample_id]["GT"]

    if None in gt_tuple:
        return {"type": "Missing", "alleles": "./."}

    decoded_alleles = [record.alleles[idx] for idx in gt_tuple]
    allele_str = "/".join(decoded_alleles)

    unique_indices = set(gt_tuple)
    if len(unique_indices) == 1:
        gt_type = "Homozygous Reference" if gt_tuple[0] == 0 else "Homozygous Alternate"
    elif 0 in unique_indices:
        gt_type = "Heterozygous"
    else:
        gt_type = "Compound Heterozygous"

    return {"type": gt_type, "alleles": allele_str}


def iter_variants(
    vcf_path: Path,
    fasta_path: Path = DEFAULT_FASTA,
    region: str | None = None,
    *,
    skip_ref_blocks: bool = True,
) -> Generator[dict[str, Any]]:
    """Yield validated variants from a VCF, one record at a time.

    Each variant is validated against ``fasta_path``: records whose REF doesn't
    match the FASTA at that position are logged and skipped (this catches
    build mismatches — e.g. supplying a GRCh37 VCF with a GRCh38 FASTA).

    Args:
        vcf_path: Path to the VCF (gzipped or not).
        fasta_path: Reference genome FASTA (must be indexed; .fai needed).
        region: Optional pysam region string (e.g. ``"chr1:1000-2000"``).
        skip_ref_blocks: If True, skip records with no ALT (gVCF reference
            blocks). Set False to retain them.

    Yields:
        ``{chrom, pos, ref, alts, quality, genotype, zygosity}`` per variant.

    Raises:
        FileNotFoundError: VCF or FASTA missing.
        BioinformaticsError: VCF has no samples, or pysam fails to open.
    """
    if not vcf_path.exists():
        msg = f"VCF not found: {vcf_path}"
        raise FileNotFoundError(msg)
    if not fasta_path.exists():
        msg = f"FASTA not found: {fasta_path}"
        raise FileNotFoundError(msg)

    try:
        vcf = pysam.VariantFile(str(vcf_path))
        genome = pysam.FastaFile(str(fasta_path))
    except OSError as e:
        msg = f"failed to open VCF/FASTA: {e}"
        raise BioinformaticsError(msg) from e

    try:
        samples: Iterable[str] = vcf.header.samples
        sample_list = list(samples)
        if not sample_list:
            msg = f"VCF has no sample columns: {vcf_path}"
            raise BioinformaticsError(msg)
        sample_id = sample_list[0]
        logger.info("Analyzing sample %s (%s)", sample_id, vcf_path.name)

        iterator = vcf.fetch(region=region) if region else vcf

        for record in iterator:
            if skip_ref_blocks and not record.alts:
                continue

            try:
                ref_from_fasta = genome.fetch(
                    record.chrom, record.start, record.stop
                ).upper()
            except KeyError:
                logger.warning("Contig %s not found in FASTA — skipping.", record.chrom)
                continue

            if ref_from_fasta != record.ref:
                logger.warning(
                    "REF mismatch at %s:%s — VCF=%s FASTA=%s — skipping.",
                    record.chrom,
                    record.pos,
                    record.ref,
                    ref_from_fasta,
                )
                continue

            info_gt = decode_genotype(record, sample_id)

            yield {
                "chrom": record.chrom,
                "pos": record.pos,
                "ref": record.ref,
                "alts": record.alts,
                "quality": record.qual,
                "genotype": info_gt["alleles"],
                "zygosity": info_gt["type"],
            }
    finally:
        vcf.close()
        genome.close()
