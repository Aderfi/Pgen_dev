"""Streaming parser for NCBI dbSNP E-utilities ESummary XML.

Turns an ``esummary`` (``db=snp``) response into structured ``DbSnpSummary``
models, one per ``<DocumentSummary>``. The reader uses ``iterparse`` and
clears each record after yielding, so memory stays flat regardless of file
size (the real dumps run to tens of MB / thousands of records).

Field provenance, per record:
    * ``SNP_ID``      -> ``snp_id`` / ``rsid``
    * ``CHR``         -> ``chromosome`` (normalized)
    * ``GENES``       -> ``genes`` (symbol + Entrez ``GENE_ID``)
    * ``FXN_CLASS``   -> ``functional_classes`` (comma-split)
    * ``SPDI``        -> ``spdi`` (0-based ``acc:pos:del:ins`` -> 1-based model)
    * ``DOCSUM``      -> ``hgvs`` (the ``HGVS=`` segment of the pipe-blob)

SPDI is preferred over DOCSUM for coordinates because it is already a clean,
machine-readable canonical form; DOCSUM is only mined for HGVS strings.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING
from xml.etree import ElementTree as ET

from src.domain.schemas.dbsnp import (
    DbSnpGene,
    DbSnpSummary,
    SpdiAllele,
    build_from_accession,
)
from src.domain.schemas.variant import normalize_chromosome

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)


def iter_summaries(path: str | Path) -> Iterator[DbSnpSummary]:
    """Yield one ``DbSnpSummary`` per ``<DocumentSummary>`` in the file.

    Malformed records are logged and skipped rather than aborting the whole
    stream — a single bad rs-id should not sink a 4000-record dump.
    """
    path = Path(path)
    context = ET.iterparse(str(path), events=("end",))
    for _event, elem in context:
        if elem.tag != "DocumentSummary":
            continue
        try:
            summary = _build_summary(elem)
        except (ValueError, TypeError) as exc:
            uid = elem.get("uid", "?")
            logger.warning("Skipping malformed dbSNP record (uid=%s): %s", uid, exc)
        else:
            if summary is not None:
                yield summary
        finally:
            elem.clear()


def parse_summaries(path: str | Path) -> list[DbSnpSummary]:
    """Eager variant of :func:`iter_summaries` (whole file into a list)."""
    return list(iter_summaries(path))


# --------------------------------------------------------------------------- #
# Record assembly                                                              #
# --------------------------------------------------------------------------- #
def _build_summary(elem: ET.Element) -> DbSnpSummary | None:
    snp_id_text = _text(elem, "SNP_ID")
    if not snp_id_text:
        return None

    return DbSnpSummary(
        snp_id=int(snp_id_text),
        chromosome=_text(elem, "CHR"),
        genes=tuple(_parse_genes(elem)),
        functional_classes=_split_csv(_text(elem, "FXN_CLASS")),
        spdi=tuple(_parse_spdi(_text(elem, "SPDI"))),
        hgvs=_parse_docsum_hgvs(_text(elem, "DOCSUM")),
    )


def _parse_genes(elem: ET.Element) -> Iterator[DbSnpGene]:
    for gene_e in elem.findall("./GENES/GENE_E"):
        name = _text(gene_e, "NAME")
        if not name:
            continue
        gene_id = _text(gene_e, "GENE_ID")
        yield DbSnpGene(
            symbol=name,
            entrez_id=int(gene_id) if gene_id and gene_id.isdigit() else None,
        )


def _parse_spdi(raw: str | None) -> Iterator[SpdiAllele]:
    """Parse the comma-separated ``acc:pos:deleted:inserted`` SPDI list.

    SPDI positions are 0-based; we store 1-based (``pos + 1``) to honor the
    project coordinate convention.
    """
    if not raw:
        return
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.split(":")
        if len(parts) != 4:
            logger.warning("Skipping malformed SPDI token %r", token)
            continue
        accession, pos0, deleted, inserted = parts
        if not pos0.lstrip("-").isdigit():
            logger.warning("Skipping SPDI token with non-numeric position %r", token)
            continue
        yield SpdiAllele(
            accession=accession,
            chrom=normalize_chromosome(accession),
            pos=int(pos0) + 1,
            deleted=deleted,
            inserted=inserted,
            build=build_from_accession(accession),
        )


def _parse_docsum_hgvs(raw: str | None) -> tuple[str, ...]:
    """Extract HGVS expressions from the ``DOCSUM`` pipe-delimited blob.

    Shape: ``HGVS=<csv>|SEQ=[..]|LEN=1|GENE=CASP7:840``. Only the ``HGVS=``
    segment is mined; the rest is redundant with structured fields.
    """
    if not raw:
        return ()
    for segment in raw.split("|"):
        key, sep, value = segment.partition("=")
        if sep and key.strip() == "HGVS":
            return tuple(item.strip() for item in value.split(",") if item.strip())
    return ()


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _text(elem: ET.Element, tag: str) -> str | None:
    child = elem.find(tag)
    if child is None or child.text is None:
        return None
    text = child.text.strip()
    return text or None


def _split_csv(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    return tuple(item.strip() for item in raw.split(",") if item.strip())


__all__ = [
    "iter_summaries",
    "parse_summaries",
]
