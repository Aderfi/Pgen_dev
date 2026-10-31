"""Per-haplotype PGx function (the path-level functional signal).

CPIC/PharmVar allele **function** (no / decreased / normal / increased + activity
score) is a property of the *star allele*, not of any single variant — so in the
per-gene graph it attaches to the **path**, not a node. This provider maps
``(gene, allele)`` to that 6-dim vector, keyed by the star-allele label directly
(``data/dicts/star_alleles.tsv``) — **no rsID** lookup, removing the old
rsID-based keying. Sub-alleles inherit their core allele's function
(``*4.001`` → ``*4`` → allele ``4``).

Vector layout (``PATH_FUNCTION_DIM`` = 6):
    function status one-hot (4) + activity_score + pgx_known mask.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

FUNC_STATUSES: tuple[str, ...] = (
    "no_function",
    "decreased_function",
    "normal_function",
    "increased_function",
)
PATH_FUNCTION_DIM: int = len(FUNC_STATUSES) + 2  # status one-hot + activity + known = 6
_STATUS_INDEX: dict[str, int] = {s: i for i, s in enumerate(FUNC_STATUSES)}
# CPIC-style activity score per function status, used when the star-allele
# row carries no explicit "AS x.x" note.
_FUNC_DEFAULT_AS: dict[str, float] = {
    "no_function": 0.0,
    "decreased_function": 0.5,
    "normal_function": 1.0,
    "increased_function": 1.5,
}
_AS_RE = re.compile(r"AS\s+([0-9]+(?:\.[0-9]+)?)")


def parse_activity_score(notes: str | None, function: str) -> float:
    """Activity score from a star-allele ``notes`` cell, else the function default.

    Parses an ``"AS x.x"`` token (``"AS 1.0+"`` → 1.0) when present; otherwise
    falls back to the CPIC-style default for the function status.
    """
    if notes:
        match = _AS_RE.search(notes)
        if match:
            return float(match.group(1))
    return _FUNC_DEFAULT_AS.get(function, 1.0)


def _core_allele(label: str) -> str:
    """``*4.001`` → ``4`` (strip the star and the sub-allele suffix)."""
    return label.lstrip("*").split(".", maxsplit=1)[0]


class HaplotypeFunctionProvider:
    """Map ``(gene, allele-label)`` to its 6-dim PGx function vector."""

    def __init__(
        self, function_by_key: dict[tuple[str, str], tuple[str, float]]
    ) -> None:
        self._function = function_by_key
        self.misses = 0

    @classmethod
    def null(cls) -> HaplotypeFunctionProvider:
        """A provider with no entries — every lookup returns a zero vector."""
        return cls({})

    @classmethod
    def from_tsv(cls, star_alleles_tsv: Path) -> HaplotypeFunctionProvider:
        """Build from ``data/dicts/star_alleles.tsv`` (gene, allele, function, notes)."""
        table = pl.read_csv(star_alleles_tsv, separator="\t", infer_schema_length=0)
        mapping: dict[tuple[str, str], tuple[str, float]] = {}
        for row in table.iter_rows(named=True):
            gene = str(row.get("gene") or "").strip()
            allele = str(row.get("allele") or "").strip()
            function = str(row.get("function") or "").strip()
            if not gene or not allele or function not in _STATUS_INDEX:
                continue
            activity = parse_activity_score(row.get("notes"), function)
            mapping.setdefault((gene, allele), (function, activity))
        logger.info(
            "HaplotypeFunction: loaded %d (gene, allele) functions from %s",
            len(mapping),
            star_alleles_tsv,
        )
        return cls(mapping)

    def vector_for(self, gene: str, label: str) -> list[float]:
        """Return the ``PATH_FUNCTION_DIM`` vector for a star-allele label.

        Misses (unknown gene/allele, or a sub-allele whose core isn't catalogued)
        yield zeros with the ``pgx_known`` mask at 0, and are tallied.
        """
        vec = [0.0] * PATH_FUNCTION_DIM
        hit = self._function.get((gene, _core_allele(label)))
        if hit is None:
            self.misses += 1
            return vec
        status, activity = hit
        vec[_STATUS_INDEX[status]] = 1.0
        vec[len(FUNC_STATUSES)] = activity
        vec[len(FUNC_STATUSES) + 1] = 1.0  # pgx_known mask
        return vec


__all__ = [
    "PATH_FUNCTION_DIM",
    "HaplotypeFunctionProvider",
    "parse_activity_score",
]
