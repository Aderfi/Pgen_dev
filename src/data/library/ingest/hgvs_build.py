"""Build a canonical genomic HGVS (``g.``) expression from VCF-style coordinates.

The genotype tower keys every variant by genomic HGVS on the RefSeq ``NC_``
accession. This module turns a ``(chrom, pos, ref, alt)`` tuple — VCF convention,
1-based, with the shared anchor base on indels — into ``NC_…:g.…``.

Normalisation here is the **minimal representation**: trim the common suffix then
the common prefix of ``ref``/``alt`` (advancing the start coordinate), which is
reference-free because VCF already carries the anchor base. This collapses
e.g. ``REF=AC, ALT=A`` to a clean ``g.{p+1}del``. The remaining HGVS subtlety —
**3'-shifting** an indel to its most-3' position inside a tandem repeat — needs
the reference sequence and is a documented later refinement; sources that already
ship a normalised HGVS (VEP, PharmVar) bypass this builder entirely.
"""

from __future__ import annotations

from src.core.exceptions import BioinformaticsError
from src.data.library.chromosome import to_refseq


def _trim(pos: int, ref: str, alt: str) -> tuple[int, str, str]:
    """Reduce ``ref``/``alt`` to their minimal representation.

    Trims the shared suffix (no coordinate change) then the shared prefix
    (advancing ``pos`` for each removed base). Either string may end empty
    (pure insertion / deletion).
    """
    # Suffix: drop equal trailing bases while both still have one.
    while ref and alt and ref[-1] == alt[-1]:
        ref, alt = ref[:-1], alt[:-1]
    # Prefix: drop equal leading bases, advancing the start coordinate.
    while ref and alt and ref[0] == alt[0]:
        ref, alt, pos = ref[1:], alt[1:], pos + 1
    return pos, ref, alt


def genomic_hgvs_body(pos: int, ref: str, alt: str) -> str:
    """Return the ``g.``-body (no accession) for a VCF-style variant.

    Raises :class:`BioinformaticsError` when ``ref == alt`` (not a variant) or the
    alleles are empty.
    """
    ref, alt = ref.upper(), alt.upper()
    if not ref or not alt:
        msg = f"empty REF/ALT (ref={ref!r}, alt={alt!r})"
        raise BioinformaticsError(msg)
    if ref == alt:
        msg = f"REF equals ALT at pos {pos} ({ref!r}) — not a variant"
        raise BioinformaticsError(msg)

    start, r, a = _trim(pos, ref, alt)

    if len(r) == 1 and len(a) == 1:  # substitution
        return f"g.{start}{r}>{a}"
    if not a:  # deletion of r at [start, start+len(r)-1]
        end = start + len(r) - 1
        span = f"{start}" if start == end else f"{start}_{end}"
        return f"g.{span}del"
    if not r:  # insertion of a between start-1 and start
        return f"g.{start - 1}_{start}ins{a}"
    # delins: r -> a at [start, start+len(r)-1]
    end = start + len(r) - 1
    span = f"{start}" if start == end else f"{start}_{end}"
    return f"g.{span}delins{a}"


def genomic_hgvs(chrom: str, pos: int, ref: str, alt: str) -> str:
    """Return the full ``NC_…:g.…`` expression for a VCF-style variant.

    ``chrom`` is resolved to its RefSeq accession via
    :func:`src.data.library.chromosome.to_refseq`.
    """
    accession = to_refseq(chrom)
    return f"{accession}:{genomic_hgvs_body(pos, ref, alt)}"


__all__ = ["genomic_hgvs", "genomic_hgvs_body"]
