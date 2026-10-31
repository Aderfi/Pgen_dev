"""Classify PharmGKB ``Variant/Haplotypes`` labels.

The label itself is kept verbatim; classification only decides which
resolver may handle it and, for labels that can never resolve to a set of
small variants, why not:

* ``rsid`` — dbSNP RefSNP.
* ``star`` — a star allele defined by small variants (PharmVar / ClinPGx).
* ``hla`` — HLA alleles are sequence-typed, not variant sets.
* ``cnv`` — copy-number and hybrid alleles (CYP2D6\\*5 deletion, ``xN``
  duplications, CYP2D6-2D7 hybrids, GSTM1/GSTT1 null).
* ``phenotype`` — metabolizer / acetylator / deficiency labels: the result of
  a diplotype, not a variant.
* ``other`` — named variants this phase has no source for (G6PD Canton,
  SLC6A4 HTTLPR repeat forms, …).
"""

from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class AlleleKind(StrEnum):
    RSID = "rsid"
    STAR = "star"
    HLA = "hla"
    CNV = "cnv"
    PHENOTYPE = "phenotype"
    OTHER = "other"


_RSID = re.compile(r"^rs(\d+)$")
_STAR = re.compile(r"^[A-Z][A-Z0-9-]*\*\S+$")
_COPY_NUMBER = re.compile(r"\*\d+[A-Z]?x(N|\d+)$")
# Structural star alleles (PharmVar structural-variation catalogue): deletions
# and hybrids with a neighbouring pseudogene/paralog are not small-variant sets.
#   CYP2D6: *5 whole-gene deletion; *13/*36/*61/*63/*68 CYP2D6-CYP2D7 hybrids.
#   CYP2A6: *4 whole-gene deletion; *12/*34 CYP2A6-CYP2A7 hybrids.
_STRUCTURAL_STARS: dict[str, frozenset[str]] = {
    "CYP2D6": frozenset({"*5", "*13", "*36", "*61", "*63", "*68"}),
    "CYP2A6": frozenset({"*4", "*12", "*34"}),
}
_NULL_GENOTYPE = re.compile(r"\b(null|non-null)$", re.IGNORECASE)
_PHENOTYPE = re.compile(
    r"metaboli[sz]er|acetylator|deficien|activity|function|phenotype|genotype",
    re.IGNORECASE,
)


class ClassifiedLabel(BaseModel):
    model_config = ConfigDict(frozen=True)

    label: str
    kind: AlleleKind
    rsid: int | None = None
    haplotype_name: str | None = None
    reason: str | None = None


def _unresolvable_kind(text: str) -> tuple[AlleleKind, str]:
    """Kind + reason for a label that is neither an rsID nor a plain star allele."""
    if text.startswith("HLA-"):
        return AlleleKind.HLA, "structural_allele"
    if _STAR.match(text) or _NULL_GENOTYPE.search(text):
        return AlleleKind.CNV, "structural_allele"
    if _PHENOTYPE.search(text):
        return AlleleKind.PHENOTYPE, "not_a_variant"
    return AlleleKind.OTHER, "not_in_pharmvar_or_tables"


def _is_structural_star(text: str) -> bool:
    gene, allele = text.split("*", 1)
    return bool(_COPY_NUMBER.search(text)) or (
        f"*{allele}" in _STRUCTURAL_STARS.get(gene, frozenset())
    )


def classify_label(label: str) -> ClassifiedLabel:
    text = label.strip()
    if m := _RSID.match(text):
        return ClassifiedLabel(label=label, kind=AlleleKind.RSID, rsid=int(m.group(1)))
    if (
        _STAR.match(text)
        and not text.startswith("HLA-")
        and not _is_structural_star(text)
    ):
        return ClassifiedLabel(label=label, kind=AlleleKind.STAR, haplotype_name=text)
    kind, reason = _unresolvable_kind(text)
    return ClassifiedLabel(label=label, kind=kind, reason=reason)


__all__ = ["AlleleKind", "ClassifiedLabel", "classify_label"]
