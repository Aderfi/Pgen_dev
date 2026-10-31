"""Static HTML report and scenario comparison, rendered from the data files."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime

import polars as pl
import pytest

from src.evaluation.compare import ComparisonReport, compare
from src.evaluation.diagnostics import OverlapReport
from src.evaluation.report import render
from src.evaluation.schemas import (
    BaselineResult,
    ClassMetrics,
    EvaluationReport,
    FoldResult,
    MetricCI,
    PairedResult,
    Spread,
    StratumResult,
)

SECTIONS = ["1 ·", "3 ·", "4 ·", "5 ·", "6 ·", "7 ·", "8 ·"]


def _ci(v: float | None) -> MetricCI:
    return MetricCI(
        value=v,
        low=None if v is None else v - 0.05,
        high=None if v is None else v + 0.05,
    )


def _class(label: str, reliable: bool) -> ClassMetrics:
    names = (
        "precision recall specificity f1 f2 f05 roc_auc pr_auc mcc brier kappa "
        "pearson_r pearson_p spearman_rho spearman_p"
    ).split()
    return ClassMetrics(
        label=label,
        support=12 if reliable else 3,
        prevalence=0.2,
        tp=5,
        fp=2,
        fn=3,
        tn=10,
        threshold=0.4,
        reliable=reliable,
        **{n: _ci(0.5) for n in names},
    )


def _report(scenario: str, f1: float, prevalence_f1: float) -> EvaluationReport:
    agg = {
        "f1_macro": _ci(f1),
        "f1_micro": _ci(f1),
        "f2_macro": _ci(f1),
        "pr_auc_macro": _ci(f1),
    }
    base = {
        "f1_macro": _ci(prevalence_f1),
        "f1_micro": _ci(prevalence_f1),
        "f2_macro": _ci(prevalence_f1),
        "pr_auc_macro": _ci(prevalence_f1),
    }
    paired = {
        k: PairedResult(diff=_ci(0.1), p_value=0.01)
        for k in ("f1_macro", "f2_macro", "pr_auc_macro")
    }
    overlap = OverlapReport(
        n_test=5,
        exact_duplicates=0,
        shared_pmids=0,
        shared_pairs=0,
        shared_drugs=1,
        shared_variants=0,
        shared_genes=0,
        exact_duplicates_pct=0,
        shared_pmids_pct=0,
        shared_pairs_pct=0,
        shared_drugs_pct=20,
        shared_variants_pct=0,
        shared_genes_pct=0,
        violations=["shared_drugs"],
    )
    return EvaluationReport(
        model="TwoTowerGAT",
        scenario=scenario,
        k=5,
        folds_evaluated=[0, 1],
        data_sha256="d",
        split_sha256="s",
        seed=711,
        n_bootstrap=100,
        created_at=datetime.now(UTC),
        n_test=5,
        rows_never_tested=0,
        rows_dropped=0,
        labels=["Toxicity", "PD"],
        headline_labels=["Toxicity"],
        aggregate=agg,
        aggregate_at_05=agg,
        aggregate_all_classes=agg,
        between_folds={"f1_macro": Spread(mean=f1, sd=0.02, n=2)},
        per_fold=[FoldResult(fold=0, n_test=3, aggregate={"f1_macro": f1})],
        per_class=[_class("Toxicity", True), _class("PD", False)],
        calibration={
            "ece_before": 0.1,
            "ece_after": 0.05,
            "mce_before": 0.2,
            "mce_after": 0.1,
            "brier_before": 0.2,
            "brier_after": 0.18,
        },
        baselines=[
            BaselineResult(name=n, aggregate=base, vs_model=paired)
            for n in ("prevalence", "drug_only", "gene_only")
        ],
        novelty=[StratumResult(flag="seen_drug", value="False", n=5, f1_macro=_ci(f1))],
        tanimoto=[
            StratumResult(
                flag="max_tanimoto", value="no structure", n=5, f1_macro=_ci(f1)
            )
        ],
        overlap=[overlap],
    )


def _write(dir_, report: EvaluationReport) -> None:
    dir_.mkdir(parents=True)
    (dir_ / "metrics.json").write_text(report.model_dump_json())
    pl.DataFrame(
        {
            "row_id": [str(i) for i in range(5)],
            "fold": [0] * 5,
            "true_labels": ["Toxicity"] * 5,
            "pred_labels": ["Toxicity", "", "Toxicity", "", "Toxicity"],
            "confidence": [0.9, 0.1, 0.5, 0.2, 0.7],
            "confidence_band": ["high", "low", "medium", "low", "high"],
            "exact_match": [True, False, True, False, True],
            "seen_drug": [False] * 5,
            "max_tanimoto": [None] * 5,
            "p_Toxicity": [0.9, 0.1, 0.6, 0.2, 0.8],
        }
    ).write_parquet(dir_ / "predictions.parquet")
    curve = {
        "pr_precision": [1, 0.5],
        "pr_recall": [0, 1],
        "roc_fpr": [0, 1],
        "roc_tpr": [0, 1],
        "rel_mean_pred": [0.05, 0.95],
        "rel_frac_pos": [0.0, 1.0],
        "rel_count": [2, 3],
    }
    (dir_ / "curves.json").write_text(json.dumps({"Toxicity": curve}))


def test_render_has_sections_in_order(tmp_path) -> None:
    _write(tmp_path / "e", _report("drug_cold", 0.4, 0.25))
    html = render(tmp_path / "e").read_text()
    positions = [html.index(f"<h2>{s}") for s in SECTIONS]
    assert positions == sorted(positions)
    assert "<script src=" not in html
    assert not re.search(r"<link[^>]+href=\"http", html)
    assert "data:image/png;base64," in html


def test_render_marks_unreliable_class_and_violations(tmp_path) -> None:
    _write(tmp_path / "e", _report("drug_cold", 0.4, 0.25))
    html = render(tmp_path / "e").read_text()
    assert "not reliable" in html
    assert "shared_drugs" in html


def test_compare_delta_and_lift(tmp_path) -> None:
    _write(tmp_path / "r", _report("random", 0.6, 0.3))
    _write(tmp_path / "d", _report("drug_cold", 0.4, 0.25))
    out = compare([tmp_path / "r", tmp_path / "d"], out_root=tmp_path / "cmp")
    report = ComparisonReport.model_validate_json((out / "comparison.json").read_text())
    row = next(r for r in report.rows if r.scenario == "drug_cold")
    assert row.delta_vs_random["f1_macro"] == pytest.approx(-0.2)
    assert row.lift_over_prevalence["f1_macro"] == pytest.approx(0.15)
    html = (out / "report.html").read_text()
    assert "<h2>2 ·" in html


def test_compare_without_random(tmp_path) -> None:
    _write(tmp_path / "d", _report("drug_cold", 0.4, 0.25))
    out = compare([tmp_path / "d"], out_root=tmp_path / "cmp")
    report = ComparisonReport.model_validate_json((out / "comparison.json").read_text())
    assert all(v is None for v in report.rows[0].delta_vs_random.values())
