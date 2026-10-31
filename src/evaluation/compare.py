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
from src.evaluation.report import REPORT_FILE, fmt, fmt_ci, page, table
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
    html_rows = [
        [
            row.scenario,
            str(row.n_test),
            f"{len(row.folds_evaluated)}/{row.k}",
            *(fmt_ci(getattr(row, k)) for k in COMPARED),
            *(
                f"{fmt(row.between_folds[k].mean)} ± {fmt(row.between_folds[k].sd)}"
                if k in row.between_folds
                else "—"
                for k in ("f1_macro",)
            ),
            *(fmt(row.delta_vs_random[k]) for k in ("f1_macro", "pr_auc_macro")),
            *(fmt(row.lift_over_prevalence[k]) for k in ("f1_macro", "pr_auc_macro")),
        ]
        for row in rows
    ]
    body = (
        "<h1>Scenario comparison</h1><h2>2 · Generalization and leakage: split scenarios</h2>"
        "<p class='note'>A large drop against <i>random</i> means that metric was inflated by "
        "leakage. Each scenario has its own test distribution, so compare the lift over its "
        "prevalence baseline too.</p>"
        + table(
            [
                "scenario",
                "n test",
                "folds",
                *COMPARED,
                "F1 macro between folds",
                "Δ F1 macro vs random",
                "Δ PR-AUC macro vs random",
                "lift F1 macro over prevalence",
                "lift PR-AUC macro over prevalence",
            ],
            html_rows,
        )
    )
    (out / REPORT_FILE).write_text(page("Scenario comparison", body))
    logger.info("Comparison of %d evaluations written to %s", len(reports), out)
    return out


__all__ = ["ComparisonReport", "ComparisonRow", "compare"]
