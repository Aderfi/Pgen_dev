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
from src.domain.schemas.hgvs import MolecularType
from src.genomics.hgvs_parser import parse


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


# Tokens PharmVar (and VEP) use for an absent allele on an indel.
_EMPTY_ALLELE = frozenset({"", "-", "."})


def hgvs_body_from_alleles(start: int, stop: int, ref: str, alt: str) -> str:
    """Return the ``g.``-body from explicit-allele coordinates (PharmVar style).

    Unlike :func:`genomic_hgvs_body` (VCF convention, shared anchor base, trimmed),
    here the alleles are already the *actual* changed bases with **no** anchor: an
    absent allele is written ``-`` (deletion has no ``alt``; insertion has no
    ``ref``, with ``start``/``stop`` the two flanking positions). Raises
    :class:`BioinformaticsError` when both alleles are empty.
    """
    r = "" if ref.strip() in _EMPTY_ALLELE else ref.strip().upper()
    a = "" if alt.strip() in _EMPTY_ALLELE else alt.strip().upper()
    if not r and not a:
        msg = f"both alleles empty at {start}"
        raise BioinformaticsError(msg)

    span = f"{start}" if start == stop else f"{start}_{stop}"
    if r and a and len(r) == 1 and len(a) == 1 and start == stop:  # substitution
        return f"g.{start}{r}>{a}"
    if not a:  # deletion over [start, stop]
        return f"g.{span}del"
    if not r:  # insertion between the flanking start and stop
        return f"g.{start}_{stop}ins{a}"
    return f"g.{span}delins{a}"  # multi-base replacement


def parse_genomic_anchor(g_hgvs: str) -> tuple[str, int, str, str]:
    """Extract ``(accession, pos, ref, alt)`` from a genomic (``g.``) HGVS string.

    The inverse of the builders: lets an adapter re-derive a consistent genomic
    anchor from a canonical ``g.`` key (``ref``/``alt`` empty for del/dup/inv,
    where the bases live in the reference). Raises :class:`BioinformaticsError`
    when the expression is not genomic or lacks an accession or a resolved start.
    """
    variant = parse(g_hgvs)
    if variant.molecular_type is not MolecularType.GENOMIC:
        msg = f"expected genomic (g.) HGVS, got {variant.molecular_type.value} in {g_hgvs!r}"
        raise BioinformaticsError(msg)
    if variant.reference_sequence is None:
        msg = f"genomic HGVS without accession: {g_hgvs!r}"
        raise BioinformaticsError(msg)

    change = variant.primary_change
    base = change.start.base if change.start is not None else None
    if base is None:
        msg = f"genomic HGVS without a resolved start position: {g_hgvs!r}"
        raise BioinformaticsError(msg)

    ref = getattr(change, "reference_allele", None) or ""
    alt = (
        getattr(change, "alternate_allele", None)
        or getattr(change, "inserted_sequence", None)
        or ""
    )
    return variant.reference_sequence, base, ref, alt


__all__ = [
    "genomic_hgvs",
    "genomic_hgvs_body",
    "hgvs_body_from_alleles",
    "parse_genomic_anchor",
]
