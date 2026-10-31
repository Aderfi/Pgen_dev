"""Canonical variant identity for the knowledge graph.

A variant is identified by its NCBI **SPDI** string on a GRCh38 RefSeq
chromosome accession, e.g. ``NC_000010.11:94781858:G:A``. SPDI positions are
0-based interbase by definition, and the string keeps that form because it is
the standard; ``CanonicalVariant.pos`` is the project's 1-based coordinate.

SPDI is only canonical after NCBI's ``canonical_representative`` service has
normalized it (fully justified). This module never normalizes indels itself.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# GRCh38 primary-assembly chromosome accessions (GRCh38.p14; chromosome
# accessions do not change across patches). RefSeq versions differ per
# chromosome, so a version threshold is wrong: chr19 is NC_000019.10 here but
# NC_000019.9 in GRCh37. Verified 2026-10-18 against NCBI nuccore titles.
GRCH38_CHROMOSOMES: frozenset[str] = frozenset(
    {
        "NC_000001.11",
        "NC_000002.12",
        "NC_000003.12",
        "NC_000004.12",
        "NC_000005.10",
        "NC_000006.12",
        "NC_000007.14",
        "NC_000008.11",
        "NC_000009.12",
        "NC_000010.11",
        "NC_000011.10",
        "NC_000012.12",
        "NC_000013.11",
        "NC_000014.9",
        "NC_000015.10",
        "NC_000016.10",
        "NC_000017.11",
        "NC_000018.10",
        "NC_000019.10",
        "NC_000020.11",
        "NC_000021.9",
        "NC_000022.11",
        "NC_000023.11",
        "NC_000024.10",
        "NC_012920.1",
    }
)
_SPDI = re.compile(
    r"^(?P<seq>[A-Z]{2}_\d+\.\d+):(?P<pos>\d+):(?P<del>[ACGTN]*):(?P<ins>[ACGTN]*)$"
)


def is_grch38_chromosome(accession: str) -> bool:
    """True for a GRCh38 primary-assembly RefSeq chromosome accession."""
    return accession in GRCH38_CHROMOSOMES


def format_spdi(seq_id: str, pos_0based: int, deleted: str, inserted: str) -> str:
    return f"{seq_id}:{pos_0based}:{deleted}:{inserted}"


def parse_spdi(spdi: str) -> tuple[str, int, str, str]:
    """Split an SPDI string into ``(seq_id, pos_0based, deleted, inserted)``."""
    match = _SPDI.match(spdi)
    if match is None:
        msg = f"not a valid SPDI expression: {spdi!r}"
        raise ValueError(msg)
    return match["seq"], int(match["pos"]), match["del"], match["ins"]


def snv_hgvs(seq_id: str, pos_1based: int, ref: str, alt: str) -> str:
    """Genomic HGVS for a single-nucleotide substitution."""
    return f"{seq_id}:g.{pos_1based}{ref}>{alt}"


class CanonicalVariant(BaseModel):
    """One allele-specific change on GRCh38, keyed by canonical SPDI."""

    model_config = ConfigDict(frozen=True)

    spdi: str
    seq_id: str
    pos: int = Field(..., ge=1, description="1-based position (SPDI position + 1).")
    ref: str = Field(..., description="Deleted sequence (SPDI 'D'); may be empty.")
    alt: str = Field(..., description="Inserted sequence (SPDI 'I'); may be empty.")
    hgvs_g: str = Field(..., min_length=1)
    assembly: Literal["GRCh38"] = "GRCh38"
    vrs_id: None = Field(
        default=None, description="Reserved for GA4GH VRS (pangenome phase)."
    )

    @field_validator("seq_id")
    @classmethod
    def _grch38_only(cls, value: str) -> str:
        if not is_grch38_chromosome(value):
            msg = f"{value!r} is not a GRCh38 chromosome accession"
            raise ValueError(msg)
        return value

    @classmethod
    def from_spdi(
        cls, seq_id: str, pos_0based: int, deleted: str, inserted: str, hgvs_g: str
    ) -> CanonicalVariant:
        return cls(
            spdi=format_spdi(seq_id, pos_0based, deleted, inserted),
            seq_id=seq_id,
            pos=pos_0based + 1,
            ref=deleted,
            alt=inserted,
            hgvs_g=hgvs_g,
        )

    @property
    def is_snv(self) -> bool:
        return len(self.ref) == 1 and len(self.alt) == 1


__all__ = [
    "GRCH38_CHROMOSOMES",
    "CanonicalVariant",
    "format_spdi",
    "is_grch38_chromosome",
    "parse_spdi",
    "snv_hgvs",
]
