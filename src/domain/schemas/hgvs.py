"""HGVS variant nomenclature — Pydantic v2 domain models.

Models the Human Genome Variation Society (HGVS) sequence-variant
nomenclature[1] at all molecular levels (genomic, coding, RNA, protein) and
supports compound expressions (cis/trans/unknown-phase), mosaicism and
chimerism.

Parsing lives in `src.genomics.hgvs_parser`; this module only describes shape.

[1] https://hgvs-nomenclature.org/stable/
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, model_validator

from src.domain.base import GenomicDomainModel
from src.domain.types import (
    MolecularType,
    ReferenceSequenceKind,
    VariantKind,
    VariantPhase,
)


# --------------------------------------------------------------------------- #
# Positions                                                                    #
# --------------------------------------------------------------------------- #
class SequencePosition(GenomicDomainModel):
    """A position on a DNA or RNA reference.

    Supports the full coding-DNA coordinate grammar:
      - `123`         → base=123
      - `-15`         → base=-15  (5'UTR)
      - `*15`         → base=15, utr3=True  (3'UTR)
      - `100+5`       → base=100, offset=+5  (intronic, downstream of exon)
      - `100-5`       → base=100, offset=-5  (intronic, upstream of exon)
      - `?`           → unknown=True
    """

    model_config = ConfigDict(frozen=True)

    base: int | None = Field(
        default=None,
        description="CDS-relative position; negative = 5'UTR. None when unknown.",
    )
    utr3: bool = Field(
        default=False,
        description="True when the position is marked with '*' (3'UTR).",
    )
    offset: int = Field(
        default=0,
        description="Intronic offset (positive = downstream of exon end).",
    )
    unknown: bool = Field(
        default=False,
        description="True when the position is the unknown marker '?'.",
    )

    @model_validator(mode="after")
    def _check_known(self) -> SequencePosition:
        if not self.unknown and self.base is None:
            msg = "SequencePosition requires `base` unless `unknown=True`"
            raise ValueError(msg)
        return self

    def __str__(self) -> str:
        if self.unknown:
            return "?"
        prefix = "*" if self.utr3 else ""
        suffix = f"{self.offset:+d}" if self.offset else ""
        return f"{prefix}{self.base}{suffix}"


# Canonical 1-letter ↔ 3-letter amino-acid code maps. `Ter`/`*` represents the
# translation stop codon; selenocysteine (Sec/U) and pyrrolysine (Pyl/O) are
# included for completeness.
AMINO_ACID_THREE_TO_ONE: dict[str, str] = {
    "Ala": "A",
    "Arg": "R",
    "Asn": "N",
    "Asp": "D",
    "Cys": "C",
    "Gln": "Q",
    "Glu": "E",
    "Gly": "G",
    "His": "H",
    "Ile": "I",
    "Leu": "L",
    "Lys": "K",
    "Met": "M",
    "Phe": "F",
    "Pro": "P",
    "Ser": "S",
    "Thr": "T",
    "Trp": "W",
    "Tyr": "Y",
    "Val": "V",
    "Sec": "U",
    "Pyl": "O",
    "Ter": "*",
}
AMINO_ACID_ONE_TO_THREE: dict[str, str] = {
    v: k for k, v in AMINO_ACID_THREE_TO_ONE.items()
}


def to_three_letter(code: str) -> str:
    """Normalize an amino-acid token to its 3-letter form.

    Accepts the canonical 3-letter code (`Gln`), the 1-letter code (`Q`),
    `Ter` or `*` for stop, and `X`/`Xaa` for an unknown residue.
    """
    if code in AMINO_ACID_THREE_TO_ONE:
        return code
    if code == "*":
        return "Ter"
    if code in {"X", "Xaa"}:
        return "Xaa"
    if len(code) == 1 and code in AMINO_ACID_ONE_TO_THREE:
        return AMINO_ACID_ONE_TO_THREE[code]
    msg = f"unknown amino-acid token: {code!r}"
    raise ValueError(msg)


class ProteinPosition(GenomicDomainModel):
    """A residue position on a protein sequence (1-based)."""

    model_config = ConfigDict(frozen=True)

    amino_acid: str = Field(
        ..., description="Reference residue (3-letter code, e.g. 'Gln' or 'Ter')."
    )
    pos: int = Field(..., ge=1, description="1-based residue index.")

    @model_validator(mode="after")
    def _normalize(self) -> ProteinPosition:
        normalized = to_three_letter(self.amino_acid)
        if normalized != self.amino_acid:
            object.__setattr__(self, "amino_acid", normalized)
        return self

    def __str__(self) -> str:
        return f"{self.amino_acid}{self.pos}"


# --------------------------------------------------------------------------- #
# Change models                                                                #
# --------------------------------------------------------------------------- #
class NucleotideChange(GenomicDomainModel):
    """A single elementary change at the DNA or RNA level."""

    model_config = ConfigDict(frozen=True)

    level: Literal["nucleotide"] = "nucleotide"
    kind: VariantKind
    start: SequencePosition
    end: SequencePosition | None = Field(
        default=None, description="End of the affected range (inclusive)."
    )
    reference_allele: str | None = Field(
        default=None, description="Bases observed on the reference (e.g. 'CTT')."
    )
    alternate_allele: str | None = Field(
        default=None, description="Substituting base for `kind=substitution`."
    )
    inserted_sequence: str | None = Field(
        default=None,
        description=(
            "Sequence inserted by ins/delins, the duplicated stretch for `dup`, "
            "or a foreign reference for `con` (e.g. 'NG_012232.1:g.345_567')."
        ),
    )
    repeat_unit: str | None = Field(
        default=None, description="The repeated unit (e.g. 'CAG') for `kind=repeat`."
    )
    repeat_count: int | None = Field(
        default=None,
        ge=0,
        description="Observed repeat count (the integer inside `[N]`).",
    )


class ProteinChange(GenomicDomainModel):
    """A single elementary change at the protein level."""

    model_config = ConfigDict(frozen=True)

    level: Literal["protein"] = "protein"
    kind: VariantKind
    start: ProteinPosition | None = Field(
        default=None,
        description="Start residue. None for `p.?` and `p.0`.",
    )
    end: ProteinPosition | None = Field(
        default=None, description="End residue of the affected range."
    )
    new_amino_acid: str | None = Field(
        default=None,
        description=("New residue for substitutions (3-letter, 'Ter' for stop)."),
    )
    inserted_residues: list[str] | None = Field(
        default=None,
        description="Residues inserted by ins/delins (each in 3-letter form).",
    )
    fs_new_residue: str | None = Field(
        default=None,
        description="First differing residue introduced by a frameshift.",
    )
    fs_terminator: int | None = Field(
        default=None,
        ge=1,
        description="Codon offset to new stop in a frameshift (None when '?').",
    )
    ext_offset: int | None = Field(
        default=None,
        description=("Initiator offset for N-terminal extension (`p.Met1ext-5` → -5)."),
    )
    ext_terminator: int | None = Field(
        default=None,
        description=(
            "Codon offset to new stop for C-terminal extension "
            "(`p.*110Trpext*17` → 17)."
        ),
    )
    uncertain: bool = Field(
        default=False,
        description="True when the change is parenthesized as predicted (`p.(...)`).",
    )

    @model_validator(mode="after")
    def _normalize_aa(self) -> ProteinChange:
        if self.new_amino_acid:
            normalized = to_three_letter(self.new_amino_acid)
            if normalized != self.new_amino_acid:
                object.__setattr__(self, "new_amino_acid", normalized)
        if self.fs_new_residue:
            normalized = to_three_letter(self.fs_new_residue)
            if normalized != self.fs_new_residue:
                object.__setattr__(self, "fs_new_residue", normalized)
        if self.inserted_residues:
            object.__setattr__(
                self,
                "inserted_residues",
                [to_three_letter(a) for a in self.inserted_residues],
            )
        return self


HGVSChange = Annotated[
    NucleotideChange | ProteinChange,
    Field(discriminator="level"),
]


# --------------------------------------------------------------------------- #
# Top-level variant                                                            #
# --------------------------------------------------------------------------- #
class HGVSVariant(GenomicDomainModel):
    """A parsed HGVS variant expression.

    Holds the reference accession (when present), molecular type, the phase
    relationship between elementary changes, and the list of changes. Single
    variants have exactly one element in `changes`; compound expressions can
    have any number.
    """

    raw: str = Field(..., description="The original HGVS string as supplied.")
    reference_sequence: str | None = Field(
        default=None,
        description="Accession including version, e.g. 'NC_000017.11'.",
    )
    reference_kind: ReferenceSequenceKind | None = Field(
        default=None, description="Classification of `reference_sequence`."
    )
    gene_symbol: str | None = Field(
        default=None,
        description=(
            "Gene symbol given in parentheses after the accession, e.g. "
            "'NM_004006.2(DMD):c.93+1G>T' → 'DMD'."
        ),
    )
    molecular_type: MolecularType
    phase: VariantPhase = VariantPhase.SINGLE
    changes: list[HGVSChange] = Field(default_factory=list, min_length=1)

    # Convenience accessors -------------------------------------------------- #
    @property
    def is_protein(self) -> bool:
        return self.molecular_type is MolecularType.PROTEIN

    @property
    def is_compound(self) -> bool:
        return self.phase in {
            VariantPhase.CIS,
            VariantPhase.TRANS,
            VariantPhase.UNKNOWN_PHASE,
            VariantPhase.HOMOZYGOUS,
        }

    @property
    def primary_change(self) -> NucleotideChange | ProteinChange:
        """The first change — convenient for single-change expressions."""
        return self.changes[0]

    @classmethod
    def molecular_type_description(cls, code: str | MolecularType) -> str:
        """Map an HGVS molecular-type letter to a human-readable name."""
        mapping = {
            MolecularType.GENOMIC: "genomic",
            MolecularType.CIRCULAR_GENOMIC: "circular genomic",
            MolecularType.MITOCHONDRIAL: "mitochondrial",
            MolecularType.CODING: "coding DNA",
            MolecularType.NON_CODING: "non-coding DNA",
            MolecularType.RNA: "RNA",
            MolecularType.PROTEIN: "protein",
        }
        try:
            key = MolecularType(code) if isinstance(code, str) else code
        except ValueError:
            return f"unknown ({code})"
        return mapping[key]
