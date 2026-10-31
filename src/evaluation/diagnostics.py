"""Leakage and novelty diagnostics.

* Novelty flags — for every test row, whether its pair / drug / variant /
  gene / article was already seen in train.
* Tanimoto — the highest ECFP4 similarity between a test row's drug(s) and
  any train drug (how "new" an unseen drug really is).
* Overlap — how many test rows share each kind of entity with train; for a
  grouped scenario the overlap on its own axis must be zero, otherwise the
  split leaked and that is reported as a violation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from pydantic import BaseModel

from src.evaluation.entities import ROW_ID, Scenario

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from src.data.cache import GraphCache
    from src.evaluation.entities import EntityResolver

logger = logging.getLogger(__name__)

NOVELTY_FLAGS = ("seen_pair", "seen_drug", "seen_variant", "seen_gene", "seen_pmid")
TANIMOTO_BINS = (0.0, 0.3, 0.5, 0.7, 0.9, 1.0)
ECFP_BITS = 1024  # src/data/library/drugs.py: _ECFP_BITS (tail of global_feats)
_RAW_IDENTITY = ("drugs_cid", "gene", "genotype", "alleles")

#: Overlap that must be zero for each grouped scenario.
_SCENARIO_AXIS = {
    Scenario.PAIR: "shared_pairs",
    Scenario.PMID: "shared_pmids",
    Scenario.DRUG_COLD: "shared_drugs",
    Scenario.VARIANT_COLD: "shared_variants",
    Scenario.GENE_COLD: "shared_genes",
}


class OverlapReport(BaseModel):
    """Test rows sharing an entity with train (counts and % of test rows)."""

    n_test: int
    exact_duplicates: int
    shared_pmids: int
    shared_pairs: int
    shared_drugs: int
    shared_variants: int
    shared_genes: int
    exact_duplicates_pct: float
    shared_pmids_pct: float
    shared_pairs_pct: float
    shared_drugs_pct: float
    shared_variants_pct: float
    shared_genes_pct: float
    violations: list[str]


def _axes(resolver: EntityResolver) -> dict[str, Callable]:
    return {
        "pair": resolver.pair,
        "drug": resolver.drugs,
        "variant": resolver.variants,
        "gene": resolver.genes,
        "pmid": resolver.pmid,
    }


def _seen(fn: Callable, train: pl.DataFrame, test: pl.DataFrame) -> list[bool]:
    known = {k for row in train.iter_rows(named=True) for k in fn(row)}
    return [bool(fn(row) & known) for row in test.iter_rows(named=True)]


def novelty_flags(
    resolver: EntityResolver, train: pl.DataFrame, test: pl.DataFrame
) -> pl.DataFrame:
    """``row_id`` + one boolean ``seen_*`` column per entity kind."""
    cols = {
        f"seen_{axis}": _seen(fn, train, test) for axis, fn in _axes(resolver).items()
    }
    return pl.DataFrame({ROW_ID: test[ROW_ID], **cols})


def drug_fingerprints(
    drug_ids: Iterable[str], cache: GraphCache
) -> dict[str, np.ndarray]:
    """ECFP4 bit vectors of the drugs present in the library."""
    out: dict[str, np.ndarray] = {}
    for drug_id in set(drug_ids):
        if drug_id not in cache.drug_index:
            continue
        graph = cache.get_drug(drug_id)
        feats = getattr(graph, "global_feats", None)
        if feats is None or feats.shape[-1] < ECFP_BITS:
            continue
        bits = feats.reshape(-1)[-ECFP_BITS:].numpy() > 0.5
        if bits.any():
            out[drug_id] = bits
    logger.info("Fingerprints for %d drugs", len(out))
    return out


def _tanimoto(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def max_tanimoto(
    test_drugs: list[frozenset[str]],
    train_drugs: set[str],
    fps: dict[str, np.ndarray],
) -> list[float | None]:
    """Max Tanimoto of each row's drugs against the train drugs (None: no structure)."""
    train_fps = [fps[d] for d in train_drugs if d in fps]
    out: list[float | None] = []
    for drugs in test_drugs:
        mine = [fps[d] for d in drugs if d in fps]
        if not mine or not train_fps:
            out.append(None)
            continue
        out.append(max(_tanimoto(a, b) for a in mine for b in train_fps))
    return out


def tanimoto_bin(value: float | None) -> str:
    """Label of the :data:`TANIMOTO_BINS` interval holding ``value``."""
    if value is None:
        return "no structure"
    edges = TANIMOTO_BINS
    for lo, hi in zip(edges, edges[1:], strict=False):
        if value < hi or hi == edges[-1]:
            return f"[{lo:.1f}, {hi:.1f}{']' if hi == edges[-1] else ')'}"
    return f"[{edges[-2]:.1f}, {edges[-1]:.1f}]"


def overlap_report(
    resolver: EntityResolver,
    train: pl.DataFrame,
    test: pl.DataFrame,
    scenario: Scenario,
) -> OverlapReport:
    n = len(test)
    counts: dict[str, int] = {}
    for axis, fn in _axes(resolver).items():
        name = "shared_pmids" if axis == "pmid" else f"shared_{axis}s"
        counts[name] = sum(_seen(fn, train, test))
    cols = [c for c in _RAW_IDENTITY if c in train.columns and c in test.columns]
    train_raw = set(train.select(cols).iter_rows())
    counts["exact_duplicates"] = sum(
        row in train_raw for row in test.select(cols).iter_rows()
    )
    pct = {f"{k}_pct": (100.0 * v / n if n else 0.0) for k, v in counts.items()}
    axis = _SCENARIO_AXIS.get(scenario)
    violations = [axis] if axis is not None and counts[axis] else []
    if violations:
        logger.error("Split leak in %s: %s", scenario, violations)
    return OverlapReport(n_test=n, violations=violations, **counts, **pct)


__all__ = [
    "NOVELTY_FLAGS",
    "TANIMOTO_BINS",
    "OverlapReport",
    "drug_fingerprints",
    "max_tanimoto",
    "novelty_flags",
    "overlap_report",
    "tanimoto_bin",
]
