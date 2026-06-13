"""Raw-HGVS adapter — ingest a file of genomic HGVS expressions directly.

The most direct, already-standardised input: a file of genomic (``g.``) HGVS
strings, one per line, or a TSV with an ``hgvs`` column (and optional ``gene``).
Each expression is parsed and validated by the project HGVS parser; non-genomic
levels (``c.``/``p.``) are skipped with a warning, since Fase 1 keys on ``g.``
(coding/protein projection to genomic is a later capability).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.core.exceptions import BioinformaticsError
from src.data.library.ingest.hgvs_build import parse_genomic_anchor
from src.data.library.ingest.models import IngestedVariant

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

logger = logging.getLogger(__name__)


def variant_from_genomic_hgvs(
    g_hgvs: str, *, gene: str | None = None
) -> IngestedVariant:
    """Build an IngestedVariant from one genomic HGVS string.

    Raises :class:`BioinformaticsError` (via :func:`parse_genomic_anchor`) when the
    string is not a usable genomic expression.
    """
    accession, pos, ref, alt = parse_genomic_anchor(g_hgvs)
    return IngestedVariant(
        gene=gene,
        g_hgvs=g_hgvs.strip(),
        accession=accession,
        pos=pos,
        ref=ref,  # may be "" for del/ins; g_hgvs stays canonical
        alt=alt,
    )


def iter_variants_from_hgvs(path: Path) -> Iterator[IngestedVariant]:
    """Yield IngestedVariants from a plain or ``hgvs``/``gene`` TSV file.

    Lines starting with ``#`` and blank lines are ignored. A header row whose
    columns include ``hgvs`` switches on TSV mode (with an optional ``gene``
    column); otherwise each line is treated as a single HGVS expression.
    """
    rows = [
        line.rstrip("\n")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not rows:
        return

    header = rows[0].split("\t")
    lowered = [c.strip().lower() for c in header]
    if "hgvs" in lowered:
        hgvs_i = lowered.index("hgvs")
        gene_i = lowered.index("gene") if "gene" in lowered else None
        for raw in rows[1:]:
            cols = raw.split("\t")
            if hgvs_i >= len(cols):
                continue
            gene = (
                cols[gene_i].strip()
                if gene_i is not None and gene_i < len(cols)
                else None
            )
            yield from _safe(cols[hgvs_i].strip(), gene or None)
    else:
        for raw in rows:
            yield from _safe(raw.strip(), None)


def _safe(g_hgvs: str, gene: str | None) -> Iterator[IngestedVariant]:
    """Yield a variant, or nothing (logged) when the HGVS is unusable here."""
    try:
        yield variant_from_genomic_hgvs(g_hgvs, gene=gene)
    except BioinformaticsError as exc:
        logger.warning("HGVS list: skipping %r — %s", g_hgvs, exc)


__all__ = ["iter_variants_from_hgvs", "variant_from_genomic_hgvs"]
