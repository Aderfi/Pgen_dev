"""Genomic variant models.

Coordinates are 1-based, half-open is *not* used here — `Position.pos` is the
exact 1-based coordinate that matches FASTA/VCF conventions. Anything 0-based
is a programming-internal detail and never leaves these models.
"""

from __future__ import annotations

import re
from typing import Annotated

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from src.domain.types import GenomeBuild, VariantType, Zygosity

# Accept short autosome names (1-22), sex chroms (X, Y), and mitochondrial (MT).
# Other contigs (scaffolds, alternate haplotypes) are valid but uncommon — we
# allow them through but require non-empty.
_CHROM_PREFIX = re.compile(r"^chr", re.IGNORECASE)
# Captures the trailing accession number with leading zeros stripped (greedy 0+ eats them).
_REFSEQ_HUMAN = re.compile(r"^NC_0+(\d+)\.\d+$")
# Special-case sex/mitochondrial chromosomes; autosomes pass through unchanged.
_REFSEQ_NON_AUTOSOME = {"23": "X", "24": "Y", "12920": "MT"}

def normalize_chromosome(chrom: str) -> str:
    """Normalize a chromosome label to the bare-token form used by Ensembl FASTA.

    Rules (in order):
      1. Strip any leading 'chr'/'CHR' prefix.
      2. Translate RefSeq accessions (NC_000001.11 → 1).
      3. Map mitochondrial aliases ('M', 'mtDNA') to 'MT'.
      4. Keep everything else as-is (allowing alt contigs / scaffolds).
    """
    if not chrom or not chrom.strip():
        msg = "chromosome cannot be empty"
        raise ValueError(msg)

    bare = _CHROM_PREFIX.sub("", chrom.strip())

    refseq = _REFSEQ_HUMAN.match(bare)
    if refseq:
        accession = refseq.group(1)
        bare = _REFSEQ_NON_AUTOSOME.get(accession, accession)

    if bare.upper() in {"M", "MTDNA"}:
        return "MT"

    return bare

ChromLabel = Annotated[str, BeforeValidator(normalize_chromosome)]
GenomicPosition = Annotated[int, Field(ge=1, description="1-based genomic position.")]

class Position(BaseModel):
    """A point on a reference genome.

    `pos` is **1-based** (FASTA/VCF convention). Code that needs 0-based
    indexing (e.g. for slicing) must subtract 1 explicitly at the call site.
    """

    model_config = ConfigDict(frozen=True)

    chrom: str = Field(
        ..., description="Chromosome label, normalized (no 'chr' prefix)."
    )
    pos: int = Field(..., ge=1, description="1-based genomic position.")
    build: GenomeBuild = Field(default=GenomeBuild.GRCH38)

    #@field_validator("chrom", mode="before")
    #@classmethod
    #def _normalize_chrom(cls, v: str) -> str:
    #    return normalize_chromosome(v)

    def __str__(self) -> str:
        return f"{self.build.value}:{self.chrom}:{self.pos}"


_ALLELE_PATTERN = re.compile(r"^[ACGTN]+$|^\.$|^-$")

def _normalize_allele(v: str) -> str:
    if not isinstance(v, str):
        msg = f"allele must be str, got {type(v).__name__}"
        raise TypeError(msg)
    upper = v.strip().upper()
    if not _ALLELE_PATTERN.match(upper):
        msg = f"invalid allele {v!r}: expected ACGTN/./- only"
        raise ValueError(msg)
    return upper

Allele = Annotated[
    str,
    BeforeValidator(_normalize_allele),
    Field(min_length=1, description="Reference/alternate allele (uppercase ACGTN)."),
]
RsId = Annotated[str, StringConstraints(pattern=r"^rs\d+$")]




class Variant(BaseModel):
    """A single sequence variant with reference and alternate alleles.

    Bioinformatics-convention: `ref` and `alt` are uppercase nucleotide strings
    (or '.'/'-' for absent allele). Multi-allelic sites must be split into
    one Variant per alternate.
    """

    model_config = ConfigDict(frozen=True)

    position: Position
    ref: Allele
    alt: Allele
    variant_type: VariantType = VariantType.OTHER
    rsid: RsId | None = Field(default=None, description="dbSNP rsID if known.")

    @model_validator(mode="after")
    def _infer_type_if_other(self) -> Variant:
        if self.variant_type is not VariantType.OTHER:
            return self
        # Use object.__setattr__ because the model is frozen.
        if len(self.ref) == 1 and len(self.alt) == 1 and self.ref != self.alt:
            object.__setattr__(self, "variant_type", VariantType.SNP)
        elif len(self.ref) > len(self.alt):
            object.__setattr__(self, "variant_type", VariantType.DELETION)
        elif len(self.ref) < len(self.alt):
            object.__setattr__(self, "variant_type", VariantType.INSERTION)
        elif len(self.ref) == len(self.alt) and len(self.ref) > 1:
            object.__setattr__(self, "variant_type", VariantType.MNP)
        return self


class Genotype(BaseModel):
    """A variant observed in a specific sample, with zygosity."""

    model_config = ConfigDict(frozen=True)

    variant: Variant
    zygosity: Zygosity
    sample_id: str | None = None


