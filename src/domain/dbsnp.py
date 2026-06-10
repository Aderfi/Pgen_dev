"""dbSNP E-utilities ESummary domain models.

Structured shape of a single ``<DocumentSummary>`` record from an NCBI
E-utilities ``esummary`` response (``db=snp``). Parsing lives in
``src.genomics.dbsnp_summary``; this module only describes shape.

Coordinate convention (project-wide): ``Position.pos`` and ``SpdiAllele.pos``
are **1-based**. NCBI SPDI is natively 0-based, so the parser adds 1 when it
builds these models — anything 0-based stays inside the parser.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.domain.variant import (
    GenomeBuild,
    Position,
    Variant,
    normalize_chromosome,
)

# RefSeq genomic accession → assembly. dbSNP build 157 ships GRCh38 (`.NN`
# patch on the current chromosome accession) alongside the prior GRCh37 one.
_GRCH38_CHR_ACCESSION = re.compile(r"^NC_0+\d+\.1[1-9]$")  # NC_000010.11, .12, ...
_GRCH37_CHR_ACCESSION = re.compile(r"^NC_0+\d+\.10$")  # NC_000010.10
_ALLELE_OR_EMPTY = re.compile(r"^[ACGTN]*$")


def build_from_accession(accession: str) -> GenomeBuild:
    """Infer the genome build from a RefSeq chromosome accession version.

    Defaults to GRCh38 (the summary's reported build) when the accession is
    not a recognized GRCh37 chromosome.
    """
    if _GRCH37_CHR_ACCESSION.match(accession):
        return GenomeBuild.GRCH37
    return GenomeBuild.GRCH38


class DbSnpGene(BaseModel):
    """A gene annotation as reported in a ``<GENE_E>`` element.

    ``entrez_id`` is the NCBI Entrez Gene ID (the ``<GENE_ID>`` value, e.g.
    840 for CASP7) — *not* an Ensembl ID, so it cannot go into ``Gene``.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str = Field(..., min_length=1, description="Gene symbol as reported.")
    entrez_id: int | None = Field(
        default=None, ge=1, description="NCBI Entrez Gene ID."
    )

    @field_validator("symbol", mode="before")
    @classmethod
    def _strip_symbol(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            msg = "gene symbol must be a non-empty str"
            raise ValueError(msg)
        return v.strip()


class SpdiAllele(BaseModel):
    """A single allele from the ``<SPDI>`` field, in SPDI semantics.

    SPDI describes a change as ``deleted -> inserted`` at a position. A
    substitution has one base in each; a deletion has an empty ``inserted``;
    an insertion has an empty ``deleted``. ``pos`` is stored **1-based**.
    """

    model_config = ConfigDict(frozen=True)

    accession: str = Field(..., description="RefSeq sequence accession.")
    chrom: str = Field(..., description="Normalized chromosome label.")
    pos: int = Field(..., ge=1, description="1-based position (SPDI 0-based + 1).")
    deleted: str = Field(..., description="Deleted sequence (empty for insertions).")
    inserted: str = Field(..., description="Inserted sequence (empty for deletions).")
    build: GenomeBuild = Field(default=GenomeBuild.GRCH38)

    @field_validator("deleted", "inserted", mode="before")
    @classmethod
    def _check_allele(cls, v: str) -> str:
        upper = (v or "").strip().upper()
        if not _ALLELE_OR_EMPTY.match(upper):
            msg = f"invalid SPDI allele {v!r}: expected ACGTN or empty"
            raise ValueError(msg)
        return upper

    def to_variant(self, rsid: str | None = None) -> Variant | None:
        """Convert to a project ``Variant`` when both alleles are present.

        Returns ``None`` for pure insertions/deletions (empty ref or alt),
        which the frozen ``Variant`` model cannot represent without an anchor
        base — those are left to the caller to normalize against the FASTA.
        """
        if not self.deleted or not self.inserted:
            return None
        return Variant(
            position=Position(chrom=self.chrom, pos=self.pos, build=self.build),
            ref=self.deleted,
            alt=self.inserted,
            rsid=rsid,
        )


class DbSnpSummary(BaseModel):
    """One ``<DocumentSummary>`` record, reduced to the PGx-relevant fields."""

    model_config = ConfigDict(frozen=True)

    snp_id: int = Field(..., ge=1, description="dbSNP numeric ID (without 'rs').")
    chromosome: str | None = Field(default=None, description="Normalized chromosome.")
    genes: tuple[DbSnpGene, ...] = ()
    functional_classes: tuple[str, ...] = Field(
        default=(), description="FXN_CLASS consequences (missense_variant, ...)."
    )
    spdi: tuple[SpdiAllele, ...] = ()
    hgvs: tuple[str, ...] = Field(
        default=(), description="HGVS expressions parsed out of DOCSUM."
    )

    @field_validator("chromosome", mode="before")
    @classmethod
    def _normalize_chrom(cls, v: str | None) -> str | None:
        if v is None or not str(v).strip():
            return None
        return normalize_chromosome(str(v))

    @property
    def rsid(self) -> str:
        return f"rs{self.snp_id}"

    def variants(self) -> list[Variant]:
        """All SPDI alleles convertible to project ``Variant`` models."""
        out: list[Variant] = []
        for allele in self.spdi:
            variant = allele.to_variant(self.rsid)
            if variant is not None:
                out.append(variant)
        return out
