"""End-to-end evaluation: split -> train each fold -> evaluate_scenario.

Synthetic graphs (see ``test_train_predict_e2e``): checks the protocol's
wiring and invariants, not model quality.
"""

from __future__ import annotations

import json

import polars as pl
import pytest

from tests.integration.test_train_predict_e2e import _train, sandbox  # noqa: F401

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _split_and_train(sandbox, monkeypatch, folds):  # noqa: F811
    from src.config import get_model_config
    from src.evaluation.entities import Scenario
    from src.evaluation.runs import FoldRef, create_splits

    tmp_path, rows = sandbox
    data = tmp_path / "train.tsv"
    rows.write_csv(data, separator="\t")
    create_splits(data, get_model_config("TwoTowerGAT"), [Scenario.DRUG_COLD], k=2)
    for fold in folds:
        _train(data, monkeypatch, split=FoldRef(scenario=Scenario.DRUG_COLD, fold=fold))
    return data


def _all_cis(obj):
    if isinstance(obj, dict):
        if {"value", "low", "high"} <= set(obj):
            yield obj
        for v in obj.values():
            yield from _all_cis(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _all_cis(v)


def test_evaluate_all_folds(sandbox, monkeypatch):  # noqa: F811
    from src.config import get_model_config
    from src.evaluation.entities import ROW_ID, Scenario
    from src.evaluation.runner import evaluate_scenario
    from src.evaluation.runs import FoldRef, load_fold_frames
    from src.evaluation.schemas import EvaluationReport
    from src.evaluation.splits import load_split

    data = _split_and_train(sandbox, monkeypatch, [0, 1])
    out = evaluate_scenario(
        "TwoTowerGAT", data, Scenario.DRUG_COLD, n_boot=50, device="cpu"
    )

    for name in (
        "metrics.json",
        "predictions.parquet",
        "predictions.csv",
        "curves.json",
    ):
        assert (out / name).exists()
    report = EvaluationReport.model_validate_json((out / "metrics.json").read_text())
    _, manifest = load_split(Scenario.DRUG_COLD)
    assert report.folds_evaluated == [0, 1]
    assert report.n_test + report.rows_never_tested == manifest.n_rows
    assert report.per_class
    assert {b.name for b in report.baselines} == {
        "prevalence",
        "drug_only",
        "gene_only",
    }
    assert len(report.overlap) == 2
    assert all(not o.violations for o in report.overlap)

    preds = pl.read_parquet(out / "predictions.parquet")
    assert preds[ROW_ID].n_unique() == len(preds) == report.n_test
    cfg = get_model_config("TwoTowerGAT")
    for fold in (0, 1):
        train, _, _, _ = load_fold_frames(
            data, cfg, FoldRef(scenario=Scenario.DRUG_COLD, fold=fold)
        )
        tested = set(preds.filter(pl.col("fold") == fold)[ROW_ID].to_list())
        assert not tested & set(train[ROW_ID].to_list())
    for col in (
        "confidence",
        "confidence_band",
        "exact_match",
        "seen_drug",
        "max_tanimoto",
    ):
        assert col in preds.columns

    from src.evaluation.report import render

    html = render(out).read_text()
    assert '<section id="predictions"' in html

    for ci in _all_cis(json.loads((out / "metrics.json").read_text())):
        if ci["low"] is not None and ci["high"] is not None and ci["value"] is not None:
            assert ci["low"] - 1e-9 <= ci["high"]


def test_evaluate_missing_fold(sandbox, monkeypatch):  # noqa: F811
    from src.core import EvaluationError
    from src.evaluation.entities import Scenario
    from src.evaluation.runner import evaluate_scenario
    from src.evaluation.schemas import EvaluationReport

    data = _split_and_train(sandbox, monkeypatch, [0])
    with pytest.raises(EvaluationError, match="fold1"):
        evaluate_scenario(
            "TwoTowerGAT", data, Scenario.DRUG_COLD, folds=[0, 1], n_boot=20
        )
    out = evaluate_scenario(
        "TwoTowerGAT", data, Scenario.DRUG_COLD, n_boot=20, device="cpu"
    )
    report = EvaluationReport.model_validate_json((out / "metrics.json").read_text())
    assert report.folds_evaluated == [0]
    assert report.k == 2


def test_evaluate_rejects_checkpoint_from_another_split(sandbox, monkeypatch):  # noqa: F811
    from src.config import get_model_config
    from src.core import EvaluationError
    from src.evaluation.entities import Scenario
    from src.evaluation.runner import evaluate_scenario
    from src.evaluation.runs import create_splits

    data = _split_and_train(sandbox, monkeypatch, [0])
    # Re-split (k=3): fold 0's checkpoint was trained on the old assignment.
    create_splits(data, get_model_config("TwoTowerGAT"), [Scenario.DRUG_COLD], k=3)
    with pytest.raises(EvaluationError, match="split"):
        evaluate_scenario("TwoTowerGAT", data, Scenario.DRUG_COLD, folds=[0], n_boot=10)
