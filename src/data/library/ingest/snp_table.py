"""Adapter for the project's pan-gene SNP table (``snp_data_output.tsv``).

The legacy variant table carries, per row, ``gene``, ``chr``, ``start_pos``,
``Ref_Allele``, ``Alt_Allele`` and ``FXN_CLASS`` (comma-separated Sequence
Ontology terms). It is the broad-coverage source behind the long tail of
non-PharmVar genes, and its ``FXN_CLASS`` populates the per-variant consequence
features. Each row is normalised to a genomic-HGVS :class:`IngestedVariant`
(VCF-style alleles, so reuse :func:`genomic_hgvs`).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import polars as pl

from src.core.exceptions import BioinformaticsError
from src.data.library.ingest.hgvs_build import genomic_hgvs, parse_genomic_anchor
from src.data.library.ingest.models import IngestedVariant

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

logger = logging.getLogger(__name__)

_REQUIRED = ("gene", "chr", "start_pos", "Ref_Allele", "Alt_Allele")
# Allele tokens that mean "no call" — skipped silently (not malformed rows).
_MISSING_ALLELE = frozenset({"", "N/A", "NA", "."})


def iter_variants_from_snp_table(path: Path) -> Iterator[IngestedVariant]:
    """Yield IngestedVariants from a ``snp_data_output.tsv``-style table."""
    frame = pl.read_csv(path, separator="\t", infer_schema_length=0)
    missing = [c for c in _REQUIRED if c not in frame.columns]
    if missing:
        msg = f"SNP table {path} missing columns {missing}"
        raise BioinformaticsError(msg)

    for row in frame.iter_rows(named=True):
        chrom = (row.get("chr") or "").strip()
        pos_raw = (row.get("start_pos") or "").strip()
        ref = (row.get("Ref_Allele") or "").strip()
        alt = (row.get("Alt_Allele") or "").strip()
        if not chrom or not pos_raw or ref in _MISSING_ALLELE or alt in _MISSING_ALLELE:
            continue
        try:
            g_hgvs = genomic_hgvs(chrom, int(pos_raw), ref, alt)
            accession, pos, a_ref, a_alt = parse_genomic_anchor(g_hgvs)
        except (BioinformaticsError, ValueError) as exc:
            logger.warning(
                "SNP table: skipping %s:%s %s>%s — %s", chrom, pos_raw, ref, alt, exc
            )
            continue
        fxn = row.get("FXN_CLASS") or ""
        so_terms = tuple(t.strip() for t in fxn.split(",") if t.strip())
        yield IngestedVariant(
            gene=(row.get("gene") or "").strip() or None,
            g_hgvs=g_hgvs,
            accession=accession,
            pos=pos,
            ref=a_ref,
            alt=a_alt,
            so_terms=so_terms,
        )


__all__ = ["iter_variants_from_snp_table"]
