"""Generic VCF adapter — a patient/sample VCF to normalised genomic-HGVS variants.

Reuses :func:`src.genomics.variant_val.iter_variants` (pysam) which validates each
record's REF against the reference FASTA (catching build mismatches), then maps
every ALT allele to a canonical genomic HGVS via
:func:`~src.data.library.ingest.hgvs_build.genomic_hgvs`. Gene assignment is left
to the caller (resolved by position later); ``gene`` defaults to ``None``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.core.exceptions import BioinformaticsError
from src.data.library.ingest.hgvs_build import genomic_hgvs, parse_genomic_anchor
from src.data.library.ingest.models import IngestedVariant

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

logger = logging.getLogger(__name__)


def iter_variants_from_vcf(
    vcf_path: Path,
    fasta_path: Path | None = None,
    *,
    gene: str | None = None,
    region: str | None = None,
) -> Iterator[IngestedVariant]:
    """Yield IngestedVariants from a (sample) VCF, validated against the FASTA."""
    # Imported here so the light ingest package doesn't pull pysam unless a VCF
    # is actually ingested.
    from src.genomics.variant_val import DEFAULT_FASTA, iter_variants

    fasta = fasta_path or DEFAULT_FASTA
    for record in iter_variants(vcf_path, fasta, region=region):
        chrom, pos, ref = record["chrom"], record["pos"], record["ref"]
        for alt in record["alts"] or ():
            try:
                g_hgvs = genomic_hgvs(chrom, pos, ref, alt)
                accession, apos, aref, aalt = parse_genomic_anchor(g_hgvs)
            except BioinformaticsError as exc:
                logger.warning(
                    "VCF: skipping %s:%s %s>%s — %s", chrom, pos, ref, alt, exc
                )
                continue
            yield IngestedVariant(
                gene=gene,
                g_hgvs=g_hgvs,
                accession=accession,
                pos=apos,
                ref=aref,  # may be "" for del/ins; g_hgvs stays canonical
                alt=aalt,
            )


__all__ = ["iter_variants_from_vcf"]
