"""Per-variant functional profile for the genotype tower (geno_global_feats).

Why this exists
---------------
The genotype graph encodes a variant's *topology* (reference vs alternate allele
bubble) but almost no signal about its *functional consequence* — and that
consequence is the pharmacogenetically causal variable. A CYP2D6 ``*4/*4`` patient
is a poor metaboliser not because of where the variant sits, but because the
allele has **no enzyme function**. Until now ``activity_score`` was hard-coded to
``0.5`` for every variant, so the single most causal feature carried no
information.

This module assembles a per-variant ``geno_global_feats`` vector (decoupled from
the node features, mirroring the drug tower's ``global_feats`` / ``admet_feats``)
from three complementary layers:

Layer A — PGx allele function (causal), keyed by rsID
    From ``data/dicts/star_alleles.tsv``: the CPIC/PharmVar **function status**
    (no / decreased / normal / increased) one-hot + the real **activity score**.
    The direct phenotype driver, but only covers known PGx star alleles.

Layer B — Sequence Ontology molecular consequence, from ``FXN_CLASS``
    Severity-aware multi-hot over consequence groups (see
    :mod:`src.data.library.consequence`). Local, high coverage — the same signal
    HGVS notation expresses in its ``VariantKind`` grammar.

Layer C — HGVS protein change, keyed by rsID
    Physicochemistry of the amino-acid substitution (Grantham, charge, hydropathy,
    volume, polarity, stop-gain, frameshift) parsed from the variant's protein
    HGVS expression (see :mod:`src.data.library.protein_change`). The protein
    expressions come from a cached dbSNP lookup; absent ⇒ that block stays zero.

``GENE_GLOBAL_DIM`` = 6 (A) + 13 (B) + 8 (C) = 27.

Every layer degrades gracefully — an unknown variant, a missing ``FXN_CLASS``,
or no protein expression each leave their block zero (with a mask flag where
relevant), so a build always runs with whatever signal is available.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import polars as pl
import torch

from src.data.library.consequence import CONSEQUENCE_DIM, consequence_vector
from src.data.library.protein_change import PROTEIN_CHANGE_DIM, protein_change_vector

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# --- Layer A — PGx allele function (by rsID) --- #
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
_FUNCTION_DIM = len(_FUNC_STATUSES) + 2  # status one-hot + activity + pgx_known = 6

#: Length of the assembled per-variant vector attached as ``geno_global_feats``.
GENE_GLOBAL_DIM: int = _FUNCTION_DIM + CONSEQUENCE_DIM + PROTEIN_CHANGE_DIM  # 27

_AS_RE = re.compile(r"AS\s+([0-9]+(?:\.[0-9]+)?)")
_RSID_COL, _HGVS_COL = "rsid", "hgvs_p"


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


def load_hgvs_protein_table(path: Path | None) -> dict[str, str]:
    """Map ``rsID -> protein HGVS expression`` from a cached dbSNP table.

    The table (``.parquet`` or delimited) needs an ``rsid`` and an ``hgvs_p``
    column. A missing path returns an empty map so Layer C degrades to zeros.
    """
    if path is None:
        logger.info("GenoFunc: no HGVS protein table — Layer C stays zero.")
        return {}
    if not path.exists():
        logger.warning("GenoFunc: HGVS table %s not found — Layer C stays zero.", path)
        return {}

    frame = (
        pl.read_parquet(path)
        if path.suffix == ".parquet"
        else pl.read_csv(path, separator="\t")
    )
    missing = {_RSID_COL, _HGVS_COL} - set(frame.columns)
    if missing:
        msg = f"HGVS table {path} missing columns {sorted(missing)}"
        raise KeyError(msg)

    mapping: dict[str, str] = {}
    for rsid, hgvs in zip(
        frame.get_column(_RSID_COL).to_list(),
        frame.get_column(_HGVS_COL).to_list(),
        strict=True,
    ):
        if rsid and hgvs:
            mapping.setdefault(str(rsid), str(hgvs))
    logger.info(
        "GenoFunc: loaded %d HGVS protein expressions from %s", len(mapping), path
    )
    return mapping


class GenoFuncProvider:
    """Assemble a variant's :data:`GENE_GLOBAL_DIM` functional vector.

    Layer A (PGx function) and Layer C (HGVS protein) are resolved by rsID
    (``variant_name``); Layer B (SO consequence) is computed from the variant's
    ``FXN_CLASS`` passed at lookup time. A variant with no signal in *any* layer
    yields a zero vector and is tallied in :attr:`misses`.
    """

    def __init__(
        self,
        function_by_rsid: dict[str, tuple[str, float]],
        hgvs_by_rsid: dict[str, str] | None = None,
        *,
        enabled: bool = True,
    ) -> None:
        self._function = function_by_rsid
        self._hgvs = hgvs_by_rsid or {}
        self._enabled = enabled
        self._zero = torch.zeros((1, GENE_GLOBAL_DIM), dtype=torch.float)
        self.misses = 0

    @classmethod
    def null(cls) -> GenoFuncProvider:
        """A disabled provider — every lookup returns a zero vector.

        Used for ``--skip-geno-func`` builds: the graph schema stays complete
        (``geno_global_feats`` present) but carries no functional signal at all,
        including the otherwise-free Layer B consequence block.
        """
        return cls({}, {}, enabled=False)

    @classmethod
    def from_sources(
        cls,
        star_alleles_tsv: Path,
        *,
        hgvs_table: Path | None = None,
    ) -> GenoFuncProvider:
        """Build a provider from the local PGx table + optional dbSNP HGVS table."""
        return cls(
            load_star_allele_function(star_alleles_tsv),
            load_hgvs_protein_table(hgvs_table),
        )

    def activity_for(self, variant_name: str) -> float | None:
        """Real activity score for a variant's rsID, or ``None`` if unknown.

        Used to replace the hard-coded ``0.5`` node-level ``activity_score``.
        """
        hit = self._function.get(variant_name)
        return hit[1] if hit is not None else None

    def _function_block(self, variant_name: str) -> tuple[list[float], bool]:
        block = [0.0] * _FUNCTION_DIM
        hit = self._function.get(variant_name)
        if hit is None:
            return block, False
        status, activity = hit
        block[_FUNC_STATUSES.index(status)] = 1.0
        block[len(_FUNC_STATUSES)] = activity
        block[len(_FUNC_STATUSES) + 1] = 1.0  # pgx_known mask
        return block, True

    def vector_for(self, variant_name: str, fxn_class: str | None) -> torch.Tensor:
        """Return the ``[1, GENE_GLOBAL_DIM]`` functional vector for a variant.

        Concatenates Layer A (PGx function, by rsID), Layer B (SO consequence,
        from ``fxn_class``) and Layer C (HGVS protein change, by rsID). A disabled
        provider (``null()`` / ``--skip-geno-func``) always returns zeros.
        """
        if not self._enabled:
            return self._zero.clone()
        func_block, func_hit = self._function_block(variant_name)
        cons_block = consequence_vector(fxn_class)
        prot_block = protein_change_vector(self._hgvs.get(variant_name))

        if not (func_hit or any(cons_block) or any(prot_block)):
            self.misses += 1
            return self._zero.clone()
        return torch.tensor(
            [[*func_block, *cons_block, *prot_block]], dtype=torch.float
        )


def zero_geno_func_vector() -> torch.Tensor:
    """A ``[1, GENE_GLOBAL_DIM]`` zero vector (the no-information functional input)."""
    return torch.zeros((1, GENE_GLOBAL_DIM), dtype=torch.float)


__all__ = [
    "GENE_GLOBAL_DIM",
    "GenoFuncProvider",
    "load_hgvs_protein_table",
    "load_star_allele_function",
    "parse_activity_score",
    "zero_geno_func_vector",
]
