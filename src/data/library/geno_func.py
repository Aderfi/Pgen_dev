"""Predicted functional / pathogenicity profile for the genotype tower.

Why this exists
---------------
The genotype graph encodes a variant's *topology* (reference vs alternate allele
bubble) but almost no signal about its *functional consequence* — and that
consequence is the pharmacogenetically causal variable. A CYP2D6 ``*4/*4`` patient
is a poor metaboliser not because of where the variant sits, but because the
allele has **no enzyme function**. Until now ``activity_score`` was hard-coded to
``0.5`` for every variant, so the single most causal feature carried no
information.

This module attaches a per-variant ``geno_global_feats`` vector (decoupled from
the node features, mirroring the drug tower's ``global_feats`` / ``admet_feats``)
built from two complementary layers:

Layer A — PGx allele function (causal), keyed by rsID
    From ``data/dicts/star_alleles.tsv``: the CPIC/PharmVar **function status**
    (no / decreased / normal / increased) one-hot + the real **activity score**.
    This is the direct phenotype driver but only covers known PGx star alleles.

Layer B — genome-wide pathogenicity (coverage), keyed by GRCh38 coordinate
    AlphaMissense (missense deleteriousness) and CADD PHRED, joined on
    ``(chrom, pos, ref, alt)``. Generic but high coverage, filling the long tail
    of variants outside the curated star-allele tables. Each score carries a
    "scored" mask so "no annotation" (zero) is distinguishable from "benign".

The ``GENE_GLOBAL_DIM`` (10) profile
------------------------------------
    A: func_no, func_decreased, func_normal, func_increased,   (4)
       activity_score, pgx_known                                (2)
    B: alphamissense, am_scored, cadd_norm, cadd_scored         (4)

Layer B degrades gracefully: with no AlphaMissense / CADD source the four B dims
are zero with mask 0 — a valid "no pathogenicity annotation" input — so a build
always runs with at least the local PGx-function signal. See
:meth:`GenoFuncProvider.from_sources`.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import polars as pl
import torch

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# --- Vector layout (order defines geno_global_feats) --- #
# Layer A — PGx allele function (by rsID).
_FUNC_STATUSES: tuple[str, ...] = (
    "no_function",
    "decreased_function",
    "normal_function",
    "increased_function",
)
# Default activity score per function status when the star-allele row omits an
# explicit "AS x.x" note — CPIC-style activity values.
_FUNC_DEFAULT_AS: dict[str, float] = {
    "no_function": 0.0,
    "decreased_function": 0.5,
    "normal_function": 1.0,
    "increased_function": 1.5,
}
_CADD_PHRED_CAP = 40.0  # PHRED scores are capped/normalised to [0, 1] at this value.

#: Length of the per-variant functional vector attached as ``geno_global_feats``.
GENE_GLOBAL_DIM: int = len(_FUNC_STATUSES) + 2 + 4  # A(6) + B(4) = 10

_AS_RE = re.compile(r"AS\s+([0-9]+(?:\.[0-9]+)?)")
_VARIANT_COLS = ("chrom", "pos", "ref", "alt")


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


def load_star_allele_function(star_alleles_tsv: Path) -> dict[str, tuple[str, float]]:
    """Map ``rsID -> (function_status, activity_score)`` from the star-allele TSV.

    Only **single-rsID** allele rows are used: a multi-rsID haplotype (e.g.
    ``rs1|rs2``) assigns its function to the *combination*, so propagating it to
    each individual rsID would mislabel a single-variant graph (e.g. SLCO1B1
    ``rs2306283`` alone is ``*37`` normal, not the ``*15`` haplotype). Rows whose
    function status is outside :data:`_FUNC_STATUSES` are skipped.
    """
    table = pl.read_csv(star_alleles_tsv, separator="\t")
    mapping: dict[str, tuple[str, float]] = {}
    for row in table.iter_rows(named=True):
        rsids = str(row.get("rsids") or "").strip()
        function = str(row.get("function") or "").strip()
        if not rsids or "|" in rsids or function not in _FUNC_STATUSES:
            continue
        activity = parse_activity_score(row.get("notes"), function)
        # First single-rsID row wins; later duplicates are consistent in practice.
        mapping.setdefault(rsids, (function, activity))
    logger.info(
        "GenoFunc: loaded %d single-rsID allele functions from %s",
        len(mapping),
        star_alleles_tsv,
    )
    return mapping


def _read_coord_scores(
    path: Path | None, score_col: str, *, label: str
) -> dict[tuple[str, int, str, str], float]:
    """Read a ``(chrom, pos, ref, alt) -> score`` table for a pathogenicity source.

    The file is any delimited table carrying ``chrom, pos, ref, alt`` plus
    ``score_col`` (chromosomes are normalised to bare ``"1".."22","X","Y"``).
    Missing path returns an empty map so Layer B degrades to zeros + mask 0.
    """
    if path is None:
        logger.info("GenoFunc: no %s source — that layer stays zero (mask 0).", label)
        return {}
    if not path.exists():
        logger.warning(
            "GenoFunc: %s source %s not found — skipping layer.", label, path
        )
        return {}

    sep = "\t" if path.suffix in {".tsv", ".gz", ".bgz"} else ","
    frame = pl.read_csv(path, separator=sep, infer_schema_length=10_000)
    needed = {*_VARIANT_COLS, score_col}
    missing = needed - set(frame.columns)
    if missing:
        msg = f"{label} source {path} missing columns {sorted(missing)}"
        raise KeyError(msg)

    scores: dict[tuple[str, int, str, str], float] = {}
    for row in frame.iter_rows(named=True):
        value = row[score_col]
        if value is None:
            continue
        chrom = str(row["chrom"]).removeprefix("chr")
        key = (chrom, int(row["pos"]), str(row["ref"]).upper(), str(row["alt"]).upper())
        scores[key] = float(value)
    logger.info("GenoFunc: loaded %d %s scores from %s", len(scores), label, path)
    return scores


class GenoFuncProvider:
    """Lookup from a variant to its :data:`GENE_GLOBAL_DIM` functional vector.

    Layer A is resolved by rsID (``variant_name``), Layer B by GRCh38 coordinate.
    A variant absent from *both* layers yields a zero vector (a valid "no
    functional annotation" input) and is tallied in :attr:`misses`.
    """

    def __init__(
        self,
        function_by_rsid: dict[str, tuple[str, float]],
        alphamissense: dict[tuple[str, int, str, str], float] | None = None,
        cadd: dict[tuple[str, int, str, str], float] | None = None,
    ) -> None:
        self._function = function_by_rsid
        self._alphamissense = alphamissense or {}
        self._cadd = cadd or {}
        self._zero = torch.zeros((1, GENE_GLOBAL_DIM), dtype=torch.float)
        self.misses = 0

    @classmethod
    def null(cls) -> GenoFuncProvider:
        """A provider with no entries — every lookup returns a zero vector.

        Used for ``--skip-geno-func`` builds: the graph schema stays complete
        (``geno_global_feats`` present) but carries no functional signal.
        """
        return cls({}, {}, {})

    @classmethod
    def from_sources(
        cls,
        star_alleles_tsv: Path,
        *,
        alphamissense_path: Path | None = None,
        cadd_path: Path | None = None,
    ) -> GenoFuncProvider:
        """Build a provider from the local PGx table + optional pathogenicity files."""
        return cls(
            load_star_allele_function(star_alleles_tsv),
            _read_coord_scores(
                alphamissense_path, "alphamissense", label="AlphaMissense"
            ),
            _read_coord_scores(cadd_path, "cadd_phred", label="CADD"),
        )

    def activity_for(self, variant_name: str) -> float | None:
        """Real activity score for a variant's rsID, or ``None`` if unknown.

        Used to replace the hard-coded ``0.5`` node-level ``activity_score``.
        """
        hit = self._function.get(variant_name)
        return hit[1] if hit is not None else None

    def vector_for(
        self,
        variant_name: str,
        chrom: str | None,
        pos: int | None,
        ref: str | None,
        alt: str | None,
    ) -> torch.Tensor:
        """Return the ``[1, GENE_GLOBAL_DIM]`` functional vector for a variant."""
        vec = [0.0] * GENE_GLOBAL_DIM

        # Layer A — PGx function by rsID.
        func_hit = self._function.get(variant_name)
        if func_hit is not None:
            status, activity = func_hit
            vec[_FUNC_STATUSES.index(status)] = 1.0
            vec[len(_FUNC_STATUSES)] = activity  # activity_score
            vec[len(_FUNC_STATUSES) + 1] = 1.0  # pgx_known mask

        # Layer B — pathogenicity by coordinate.
        b0 = len(_FUNC_STATUSES) + 2
        coord_hit = False
        if None not in (chrom, pos, ref, alt):
            key = (
                str(chrom).removeprefix("chr"),
                int(pos),
                str(ref).upper(),
                str(alt).upper(),
            )
            am = self._alphamissense.get(key)
            if am is not None:
                vec[b0] = am
                vec[b0 + 1] = 1.0
                coord_hit = True
            cadd = self._cadd.get(key)
            if cadd is not None:
                vec[b0 + 2] = min(cadd, _CADD_PHRED_CAP) / _CADD_PHRED_CAP
                vec[b0 + 3] = 1.0
                coord_hit = True

        if func_hit is None and not coord_hit:
            self.misses += 1
            return self._zero.clone()
        return torch.tensor([vec], dtype=torch.float)


def zero_geno_func_vector() -> torch.Tensor:
    """A ``[1, GENE_GLOBAL_DIM]`` zero vector (the no-information functional input)."""
    return torch.zeros((1, GENE_GLOBAL_DIM), dtype=torch.float)


__all__ = [
    "GENE_GLOBAL_DIM",
    "GenoFuncProvider",
    "load_star_allele_function",
    "parse_activity_score",
    "zero_geno_func_vector",
]
