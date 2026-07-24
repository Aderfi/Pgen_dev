"""Type aliases and enums shared across the domain models.

Open-ended/free-text categories are modeled as `Literal`.
Genomic codes with a cryptic short form (accession prefixes, molecular-type letters)
are `StrEnum`.
"""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field

type PositiveInt = Annotated[int, Field(gt=0)]
type NonEmptyStr = Annotated[str, Field(min_length=1)]
type PubChemCID = Annotated[int, Field(gt=0, le=10**9)]  # PubChem Compound ID:
type PubChemSID = Annotated[
    int, Field(gt=0, le=10**9)
]  # PubChem Substance ID !!!! To use only for substances that have not associated compunds

# Domain categories
FrequencyCategory = Literal[
    "very common",  # ≥1/10
    "common",  # ≥1/100 to <1/10
    "uncommon",  # ≥1/1,000 to <1/100
    "rare",  # ≥1/10,000 to <1/1,000
    "very rare",  # <1/10,000
]

SeverityLevel = Literal["mild", "moderate", "severe"]

# Imports of schemas from another project. Still designing the domain models, but they are not part of this project.
# They are used for type hinting and validation of data that is passed between different parts of the system.

# InteractionType = Literal["PD", "PK"]
# InteractionSeverity = Literal["minor", "moderate", "major", "contraindicated"]

# Provenance source names. LLM_NORMALIZED tags MedDRA codes derived by the
# offline term-normalizer (never a clinical assertion, only a coding step).
# BEERS tags age-modifier rules from the AGS Beers Criteria.

# MedDRA numeric code (Preferred/Lower Level Term id), 5 to 8 digits.
# - type MedDRACode = Annotated[str, Field(pattern=r'^\\d{5,8}$')] # noqa: W605   #type: ignore #noqa: W605

## -------------------------------------------------------------------------------------------

# Genomic Enums. (StrEnums)


class ReferenceSequenceKind(StrEnum):
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


class MolecularType(StrEnum):
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


class VariantKind(StrEnum):
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


class VariantPhase(StrEnum):
    """Relationship between multiple changes in a single HGVS expression."""

    SINGLE = "single"  # one elementary change, no brackets
    CIS = "cis"  # [a;b]     — same allele
    TRANS = "trans"  # [a];[b]   — different alleles
    UNKNOWN_PHASE = "unknown"  # [a(;)b]   — phase not established
    MOSAIC = "mosaic"  # A=/>G     — somatic mosaic
    CHIMERIC = "chimeric"  # A=//>G    — chimeric tissue
    HOMOZYGOUS = "homozygous"  # [a];[a]   or [a](;)[a]


class GenomeBuild(StrEnum):
    """Reference assembly identifier.

    Pharmagen targets GRCh38 by default; GRCh37 is accepted but flagged at the
    Variant boundary so callers can opt in deliberately (legacy datasets).
    """

    GRCH37 = "GRCh37"
    GRCH38 = "GRCh38"


class VariantType(StrEnum):
    SNP = "SNP"
    MNP = "MNP"
    INSERTION = "INS"
    DELETION = "DEL"
    INDEL = "INDEL"
    STAR_ALLELE = "STAR_ALLELE"
    OTHER = "OTHER"


class Zygosity(StrEnum):
    HOMOZYGOUS_REF = "0/0"
    HETEROZYGOUS = "0/1"
    HOMOZYGOUS_ALT = "1/1"
    HEMIZYGOUS = "1"
    UNKNOWN = "./."
