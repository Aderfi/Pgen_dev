"""Scenario comparison: how much each "never seen" scenario costs.

Each scenario has its own test set and class distribution, so besides the
raw difference against ``random`` the comparison reports each scenario's
lift over its own prevalence baseline.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel

from src.config import get_settings
from src.evaluation import EVAL_DIRNAME
from src.evaluation.report import (
    REPORT_FILE,
    delta_chip,
    fmt,
    interval,
    page,
    section,
    stat,
    table,
)
from src.evaluation.schemas import EvaluationReport, MetricCI, Spread

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

COMPARED = ("f1_micro", "f1_macro", "f2_macro", "pr_auc_macro")


class ComparisonRow(BaseModel):
    scenario: str
    n_test: int
    folds_evaluated: list[int]
    k: int
    f1_micro: MetricCI
    f1_macro: MetricCI
    f2_macro: MetricCI
    pr_auc_macro: MetricCI
    between_folds: dict[str, Spread]
    delta_vs_random: dict[str, float | None]
    lift_over_prevalence: dict[str, float | None]


class ComparisonReport(BaseModel):
    created_at: datetime
    rows: list[ComparisonRow]


def _value(report: EvaluationReport, key: str) -> float | None:
    ci = report.aggregate.get(key)
    return ci.value if ci is not None else None


def _diff(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a - b


def _spread(row: ComparisonRow, key: str) -> str:
    s = row.between_folds.get(key)
    if s is None:
        return "—"
    return fmt(s.mean) if s.sd is None else f"{fmt(s.mean)} ± {fmt(s.sd)}"


def _metric_cell(row: ComparisonRow, key: str) -> str:
    """Value + CI, its between-folds spread, and both shifts as chips."""
    return (
        f"{interval(getattr(row, key))}"
        f'<small class="ci">folds {_spread(row, key)}</small><div class="dl">'
        f"<small>Δ random</small>"
        f"{delta_chip(row.delta_vs_random.get(key), tip=f'{key}: scenario − random')}"
        f"<small>lift</small>"
        f"{delta_chip(row.lift_over_prevalence.get(key), tip=f'{key}: model − prevalence')}"
        "</div>"
    )


def _render(result: ComparisonReport) -> str:
    """Comparison page in the evaluation report's design."""
    rows = result.rows
    stats = (
        stat("Scenarios", str(len(rows)))
        + stat("Test rows", f"{sum(r.n_test for r in rows):,}", "summed over scenarios")
        + stat(
            "Random baseline",
            "yes" if any(r.scenario == "random" for r in rows) else "no",
            "reference for Δ",
        )
        + stat("Created", f"{result.created_at:%Y-%m-%d %H:%M}", "UTC")
    )
    hero = (
        '<section id="overview" class="hero" data-nav="Overview">'
        '<span class="eyebrow">Generalization &amp; leakage</span>'
        "<h1>Scenario comparison</h1>"
        f"<p class='scn'>{' · '.join(r.scenario for r in rows)}</p>"
        f'<div class="stats">{stats}</div></section>'
    )
    body = [
        [
            f"<b>{r.scenario}</b>",
            (f' data-v="{r.n_test}"', f"{r.n_test:,}"),
            f"{len(r.folds_evaluated)}/{r.k}",
            *(_metric_cell(r, k) for k in COMPARED),
        ]
        for r in rows
    ]
    groups = [("", 3), ("Out-of-fold test · 95% CI", len(COMPARED))]
    heads = ["scenario", "n test", "folds", *COMPARED]
    return hero + section(
        "scenarios",
        "Split scenarios",
        table(heads, body, attrs='class="sticky"', groups=groups),
        "Each cell: value with 95% CI, mean ± sd between folds, Δ against <i>random</i> "
        "(a large drop means the metric was inflated by leakage) and lift over the "
        "scenario's own prevalence baseline (each scenario has its own test distribution).",
        "Scenarios",
    )


def compare(eval_dirs: list[Path], *, out_root: Path | None = None) -> Path:
    """Write ``comparison.json`` + ``report.html`` for several evaluations."""
    reports = [
        EvaluationReport.model_validate_json((d / "metrics.json").read_text())
        for d in eval_dirs
    ]
    random = next((r for r in reports if str(r.scenario) == "random"), None)
    rows = []
    for r in reports:
        prevalence = next((b for b in r.baselines if b.name == "prevalence"), None)
        rows.append(
            ComparisonRow(
                scenario=str(r.scenario),
                n_test=r.n_test,
                folds_evaluated=r.folds_evaluated,
                k=r.k,
                **{k: r.aggregate.get(k, MetricCI(value=None)) for k in COMPARED},
                between_folds={
                    k: v for k, v in r.between_folds.items() if k in COMPARED
                },
                delta_vs_random={
                    k: _diff(_value(r, k), _value(random, k)) if random else None
                    for k in COMPARED
                },
                lift_over_prevalence={
                    k: _diff(
                        _value(r, k),
                        prevalence.aggregate[k].value
                        if prevalence and k in prevalence.aggregate
                        else None,
                    )
                    for k in COMPARED
                },
            )
        )
    result = ComparisonReport(created_at=datetime.now(UTC), rows=rows)
    root = out_root or get_settings().paths.reports / EVAL_DIRNAME
    out = root / f"compare_{result.created_at:%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(result.model_dump_json(indent=2))
    (out / REPORT_FILE).write_text(page("Scenario comparison", _render(result)))
    logger.info("Comparison of %d evaluations written to %s", len(reports), out)
    return out


__all__ = ["ComparisonReport", "ComparisonRow", "compare"]
