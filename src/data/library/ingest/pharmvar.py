"""PharmVar adapter — a star allele becomes its group of co-occurring HGVS variants.

PharmVar ships, per gene, a ``<GENE>.<NC_accession>.haplotypes.tsv`` listing one
row per (haplotype, variant): ``Haplotype Name, Gene, rsID, ReferenceSequence,
Variant Start, Variant Stop, Reference Allele, Variant Allele, Type``. Grouping by
``Haplotype Name`` reconstitutes each star/sub-allele as the *set of variants that
co-occur on it* — e.g. ``CYP2D6*4.001`` spans ~16 substitutions across the gene.
This is the unit the old builder collapsed to a single position; here it becomes a
path in the gene graph.

Alleles use PharmVar's explicit convention (``-`` for an absent allele, flanking
positions for insertions), handled by
:func:`~src.data.library.ingest.hgvs_build.hgvs_body_from_alleles`. The
``ReferenceSequence`` column gives the ``NC_`` accession directly, so the genomic
HGVS key needs no chromosome mapping. rsIDs are intentionally **not** retained —
the variant is keyed by HGVS, not by rsID.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import polars as pl

from src.core.exceptions import BioinformaticsError
from src.data.library.ingest.hgvs_build import hgvs_body_from_alleles
from src.data.library.ingest.models import IngestedHaplotype, IngestedVariant

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

logger = logging.getLogger(__name__)

_HAPLO_GLOB = "*.haplotypes.tsv"
_REFERENCE_SEQ = "REFERENCE"  # ReferenceSequence value on a reference (*1) row
_NAME, _GENE, _SEQ = "Haplotype Name", "Gene", "ReferenceSequence"
_START, _STOP = "Variant Start", "Variant Stop"
_REF, _ALT = "Reference Allele", "Variant Allele"


def _strip_gene_prefix(haplotype_name: str, gene: str) -> str:
    """``CYP2D6*4.001`` → ``*4.001`` (the path label inside the gene graph)."""
    return (
        haplotype_name[len(gene) :]
        if haplotype_name.startswith(gene)
        else haplotype_name
    )


def _variant_from_row(gene: str, row: dict[str, str]) -> IngestedVariant | None:
    """Build an IngestedVariant from one TSV row, or None for reference rows."""
    accession = (row.get(_SEQ) or "").strip()
    start_raw = (row.get(_START) or "").strip()
    if not accession or accession == _REFERENCE_SEQ or start_raw in {"", "."}:
        return None  # reference / placeholder row carries no variant

    start = int(start_raw)
    stop = int((row.get(_STOP) or start_raw).strip() or start_raw)
    ref = (row.get(_REF) or "").strip()
    alt = (row.get(_ALT) or "").strip()
    try:
        body = hgvs_body_from_alleles(start, stop, ref, alt)
    except BioinformaticsError as exc:
        logger.warning("PharmVar: skipping unbuildable variant (%s): %s", row, exc)
        return None

    normal_ref = "" if ref in {"-", ".", ""} else ref
    normal_alt = "" if alt in {"-", ".", ""} else alt
    return IngestedVariant(
        gene=gene,
        g_hgvs=f"{accession}:{body}",
        accession=accession,
        pos=start,
        ref=normal_ref or normal_alt,  # keep a non-empty REF marker for the model
        alt=normal_alt or normal_ref,
    )


def load_gene_haplotypes(haplotypes_tsv: Path) -> list[IngestedHaplotype]:
    """Parse one ``<GENE>.<NC>.haplotypes.tsv`` into its haplotypes (paths)."""
    # infer_schema_length=0 → every column stays Utf8 (Start/Stop are '.' on
    # reference rows, which would otherwise break integer inference).
    frame = pl.read_csv(
        haplotypes_tsv,
        separator="\t",
        comment_prefix="#",
        infer_schema_length=0,
    )
    if _NAME not in frame.columns:
        msg = f"PharmVar haplotypes TSV missing '{_NAME}' column: {haplotypes_tsv}"
        raise BioinformaticsError(msg)

    grouped: dict[str, list[IngestedVariant]] = {}
    gene_of: dict[str, str] = {}
    for row in frame.iter_rows(named=True):
        name = (row.get(_NAME) or "").strip()
        if not name:
            continue
        gene = (row.get(_GENE) or "").strip() or name.split("*")[0]
        gene_of.setdefault(name, gene)
        grouped.setdefault(name, [])
        variant = _variant_from_row(gene, row)
        if variant is not None:
            grouped[name].append(variant)

    haplotypes = [
        IngestedHaplotype(
            gene=gene_of[name],
            label=_strip_gene_prefix(name, gene_of[name]),
            variants=tuple(sorted(variants, key=lambda v: v.pos)),
        )
        for name, variants in grouped.items()
    ]
    logger.info(
        "PharmVar: %s → %d haplotypes (%d with variants).",
        haplotypes_tsv.name,
        len(haplotypes),
        sum(1 for h in haplotypes if h.variants),
    )
    return haplotypes


def iter_haplotypes(pharmvar_dir: Path) -> Iterator[IngestedHaplotype]:
    """Yield every haplotype across a PharmVar per-gene folder tree."""
    if not pharmvar_dir.exists():
        logger.warning("PharmVar folder not found: %s", pharmvar_dir)
        return
    for tsv in sorted(pharmvar_dir.rglob(_HAPLO_GLOB)):
        yield from load_gene_haplotypes(tsv)


__all__ = ["iter_haplotypes", "load_gene_haplotypes"]
