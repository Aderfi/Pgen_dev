"""Node features for the HGVS genotype graph, read off the HGVS notation itself.

Every variant node is described by what its HGVS expression states (edit type,
molecular level, coding region, sequence change) plus the annotation the
variant catalog holds for it (Sequence Ontology consequence, protein change)
and how many copies the patient carries. A gene node carries no variant
information — its identity enters the model as a learned embedding
(``gene_token``), not as features.

Layout (``HGVS_NODE_DIM`` = 45)::

    node type        (2)  [is_gene, is_variant]
    edit kind        (8)  substitution, deletion, duplication, insertion,
                          delins, inversion, repeat, other
    molecular level  (3)  genomic (g./m./o.), transcript (c./n./r.), protein (p.)
    coding region    (5)  cds, intron (c.N+k), 5'UTR (c.-N), 3'UTR (c.*N), unknown
    sequence change  (4)  transition, transversion, log length, frameshift-like
    consequence     (13)  SO consequence (src.data.library.consequence)
    protein change   (8)  HGVS p. physicochemistry (src.data.library.protein_change)
    dosage           (2)  [alt copies / 2, copies known]
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

from src.core import BioinformaticsError
from src.data.library.consequence import CONSEQUENCE_DIM, consequence_vector
from src.data.library.protein_change import PROTEIN_CHANGE_DIM, protein_change_vector
from src.domain.types import MolecularType, VariantKind
from src.genomics.hgvs_parser import parse

if TYPE_CHECKING:
    from collections.abc import Iterable

    from src.domain.schemas.hgvs import HGVSVariant, NucleotideChange, ProteinChange

NODE_TYPES: tuple[str, ...] = ("gene", "variant")
EDIT_KINDS: tuple[str, ...] = (
    "substitution",
    "deletion",
    "duplication",
    "insertion",
    "delins",
    "inversion",
    "repeat",
    "other",
)
LEVELS: tuple[str, ...] = ("genomic", "transcript", "protein")
REGIONS: tuple[str, ...] = ("cds", "intron", "utr5", "utr3", "unknown")
CHANGE_DIM = 4
DOSAGE_DIM = 2

HGVS_NODE_DIM: int = (
    len(NODE_TYPES)
    + len(EDIT_KINDS)
    + len(LEVELS)
    + len(REGIONS)
    + CHANGE_DIM
    + CONSEQUENCE_DIM
    + PROTEIN_CHANGE_DIM
    + DOSAGE_DIM
)

_EDIT_INDEX = {k: i for i, k in enumerate(EDIT_KINDS)}
_LEVEL_OF: dict[MolecularType, str] = {
    MolecularType.GENOMIC: "genomic",
    MolecularType.CIRCULAR_GENOMIC: "genomic",
    MolecularType.MITOCHONDRIAL: "genomic",
    MolecularType.CODING: "transcript",
    MolecularType.NON_CODING: "transcript",
    MolecularType.RNA: "transcript",
    MolecularType.PROTEIN: "protein",
}
_PURINES = frozenset("AG")
# Changes longer than this many bases saturate the log-length feature.
_LENGTH_CAP = 1000


@dataclass(frozen=True)
class Dosage:
    """Copies of the alternate allele the patient carries (None = unknown)."""

    copies: int | None = None

    def vector(self) -> list[float]:
        if self.copies is None:
            return [0.0, 0.0]
        return [min(self.copies, 2) / 2.0, 1.0]


UNKNOWN_DOSAGE = Dosage()


@lru_cache(maxsize=1 << 16)
def parse_hgvs(text: str) -> HGVSVariant | None:
    """Parse ``text`` as an HGVS expression with a reference, or ``None``.

    Cached: the same few thousand variants recur across every row and epoch.
    """
    text = text.strip()
    if ":" not in text:
        return None
    try:
        return parse(text)
    except BioinformaticsError:
        return None


def _edit_kind(change: NucleotideChange | ProteinChange) -> str:
    kind = change.kind.value
    return kind if kind in _EDIT_INDEX else "other"


def _region(variant: HGVSVariant, change: NucleotideChange | ProteinChange) -> str:
    """Coding-DNA region from ``c.`` coordinates; ``unknown`` for other levels."""
    if (
        variant.molecular_type is not MolecularType.CODING
        or change.level != "nucleotide"
    ):
        return "unknown"
    pos = change.start
    if pos.unknown or pos.base is None:
        return "unknown"
    if pos.offset:
        return "intron"
    if pos.utr3:
        return "utr3"
    if pos.base < 0:
        return "utr5"
    return "cds"


def change_length(change: NucleotideChange | ProteinChange) -> int:
    """Bases (or residues) affected — span for ranges, inserted length for ins,
    and the repeated stretch for repeats (``T[5]GTT[1]`` → 8), so repeat
    alleles of different copy number differ."""
    if change.level != "nucleotide":
        return 1
    if change.kind is VariantKind.INSERTION and change.inserted_sequence:
        return len(change.inserted_sequence)
    if change.kind is VariantKind.REPEAT and change.repeat_length is not None:
        return max(change.repeat_length, 1)
    start, end = change.start, change.end
    if end is None or start.base is None or end.base is None:
        return 1
    return abs(end.base - start.base) + 1


def _sequence_change(change: NucleotideChange | ProteinChange) -> list[float]:
    vec = [0.0] * CHANGE_DIM
    if change.level == "nucleotide" and change.kind is VariantKind.SUBSTITUTION:
        ref, alt = change.reference_allele, change.alternate_allele
        if ref and alt:
            same_class = (ref.upper() in _PURINES) == (alt.upper() in _PURINES)
            vec[0 if same_class else 1] = 1.0
    length = change_length(change)
    vec[2] = math.log1p(min(length, _LENGTH_CAP)) / math.log1p(_LENGTH_CAP)
    if change.level == "nucleotide":
        indel = change.kind in (
            VariantKind.DELETION,
            VariantKind.INSERTION,
            VariantKind.DUPLICATION,
            VariantKind.DELINS,
        )
        vec[3] = 1.0 if indel and length % 3 else 0.0
    else:
        vec[3] = 1.0 if change.kind is VariantKind.FRAMESHIFT else 0.0
    return vec


def _one_hot(options: tuple[str, ...], value: str) -> list[float]:
    return [1.0 if o == value else 0.0 for o in options]


def gene_node_vector() -> list[float]:
    """Feature row of a gene hub node (identity comes from ``gene_token``)."""
    vec = [0.0] * HGVS_NODE_DIM
    vec[0] = 1.0
    return vec


def variant_node_vector(
    variant: HGVSVariant,
    change: NucleotideChange | ProteinChange,
    *,
    so_terms: Iterable[str] = (),
    p_hgvs: str | None = None,
    dosage: Dosage = UNKNOWN_DOSAGE,
) -> list[float]:
    """Feature row of one elementary change of an HGVS expression.

    ``so_terms`` / ``p_hgvs`` come from the variant catalog; a ``p.`` expression
    describes its own protein change.
    """
    if variant.molecular_type is MolecularType.PROTEIN:
        p_hgvs = p_hgvs or variant.raw
    return [
        *_one_hot(NODE_TYPES, "variant"),
        *_one_hot(EDIT_KINDS, _edit_kind(change)),
        *_one_hot(LEVELS, _LEVEL_OF[variant.molecular_type]),
        *_one_hot(REGIONS, _region(variant, change)),
        *_sequence_change(change),
        *consequence_vector(",".join(so_terms) or None),
        *protein_change_vector(p_hgvs),
        *dosage.vector(),
    ]


def alt_allele(variant: HGVSVariant) -> str | None:
    """The alternate base of a single-nucleotide substitution, else ``None``."""
    change = variant.changes[0]
    if (
        len(variant.changes) == 1
        and change.level == "nucleotide"
        and change.kind is VariantKind.SUBSTITUTION
    ):
        return (change.alternate_allele or "").upper() or None
    return None


def start_position(
    variant: HGVSVariant, change: NucleotideChange | ProteinChange
) -> int | None:
    """Start coordinate on ``variant.reference_sequence`` (``None`` if unknown)."""
    if change.level == "nucleotide":
        return None if change.start.unknown else change.start.base
    return change.start.pos if change.start is not None else None


__all__ = [
    "EDIT_KINDS",
    "HGVS_NODE_DIM",
    "LEVELS",
    "NODE_TYPES",
    "REGIONS",
    "UNKNOWN_DOSAGE",
    "Dosage",
    "alt_allele",
    "change_length",
    "gene_node_vector",
    "parse_hgvs",
    "start_position",
    "variant_node_vector",
]
