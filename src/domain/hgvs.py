"""HGVS variant nomenclature — Pydantic v2 domain models.

Models the Human Genome Variation Society (HGVS) sequence-variant
nomenclature[1] at all molecular levels (genomic, coding, RNA, protein) and
supports compound expressions (cis/trans/unknown-phase), mosaicism and
chimerism.

Parsing lives in `src.genomics.hgvs_parser`; this module only describes shape.

[1] https://hgvs-nomenclature.org/stable/
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


# --------------------------------------------------------------------------- #
# Enums                                                                        #
# --------------------------------------------------------------------------- #
class ReferenceSequenceKind(str, Enum):
    """Accession prefix of the reference sequence.

    Covers RefSeq (NC_/NG_/NM_/...) plus LRG and Ensembl ENSG/ENST/ENSP.
    """

    GENOMIC_CHROMOSOME = "NC"  # NC_000017.11 — complete chromosome
    GENOMIC_REGION = "NG"  # NG_*         — incomplete genomic region
    GENOMIC_UNPLACED = "NT"
    GENOMIC_UNLOCALIZED = "NW"
    MRNA = "NM"  # NM_007294.4  — curated mRNA / transcript
    NON_CODING_RNA = "NR"
    PROTEIN = "NP"  # NP_*         — curated protein
    PREDICTED_MRNA = "XM"
    PREDICTED_NON_CODING_RNA = "XR"
    PREDICTED_PROTEIN = "XP"
    LRG = "LRG"  # LRG_8t1      — Locus Reference Genomic
    ENSEMBL_GENE = "ENSG"
    ENSEMBL_TRANSCRIPT = "ENST"
    ENSEMBL_PROTEIN = "ENSP"
    UNKNOWN = "UNKNOWN"


class MolecularType(str, Enum):
    """The letter prefix that precedes the dot (e.g. the `c` in `c.5434C>T`)."""

    GENOMIC = "g"  # linear genomic reference
    CIRCULAR_GENOMIC = "o"  # circular genomic reference (rare)
    MITOCHONDRIAL = "m"  # mitochondrial reference
    CODING = "c"  # coding DNA (CDS-relative)
    NON_CODING = "n"  # non-coding DNA reference
    RNA = "r"  # RNA reference (lowercase bases)
    PROTEIN = "p"  # protein reference


# Convenience — which molecular types parse as nucleotide vs protein.
NUCLEOTIDE_TYPES: frozenset[MolecularType] = frozenset(
    {
        MolecularType.GENOMIC,
        MolecularType.CIRCULAR_GENOMIC,
        MolecularType.MITOCHONDRIAL,
        MolecularType.CODING,
        MolecularType.NON_CODING,
        MolecularType.RNA,
    }
)


class VariantKind(str, Enum):
    """The class of change described by the variant."""

    SUBSTITUTION = "substitution"  # A>G        (DNA) or Gln1812Arg (protein)
    DELETION = "deletion"  # del
    DUPLICATION = "duplication"  # dup
    INSERTION = "insertion"  # ins
    DELINS = "delins"  # delins / indel
    INVERSION = "inversion"  # inv
    CONVERSION = "conversion"  # con — inter-sequence copy
    REPEAT = "repeat"  # [N] — short tandem repeat count
    FRAMESHIFT = "frameshift"  # fs        (protein only)
    EXTENSION = "extension"  # ext       (protein only)
    SILENT = "silent"  # =         (no change)
    UNKNOWN = "unknown"  # ?         (effect not known)
    NO_PROTEIN = "no_protein"  # p.0       (no product translated)


class VariantPhase(str, Enum):
    """Relationship between multiple changes in a single HGVS expression."""

    SINGLE = "single"  # one elementary change, no brackets
    CIS = "cis"  # [a;b]     — same allele
    TRANS = "trans"  # [a];[b]   — different alleles
    UNKNOWN_PHASE = "unknown"  # [a(;)b]   — phase not established
    MOSAIC = "mosaic"  # A=/>G     — somatic mosaic
    CHIMERIC = "chimeric"  # A=//>G    — chimeric tissue
    HOMOZYGOUS = "homozygous"  # [a];[a]   or [a](;)[a]


# --------------------------------------------------------------------------- #
# Positions                                                                    #
# --------------------------------------------------------------------------- #
class SequencePosition(BaseModel):
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


class ProteinPosition(BaseModel):
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
class NucleotideChange(BaseModel):
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


class ProteinChange(BaseModel):
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
class HGVSVariant(BaseModel):
    """A parsed HGVS variant expression.

    Holds the reference accession (when present), molecular type, the phase
    relationship between elementary changes, and the list of changes. Single
    variants have exactly one element in `changes`; compound expressions can
    have any number.
    """

    model_config = ConfigDict(frozen=True)

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
