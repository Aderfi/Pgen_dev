"""Runner guards: fold resolution, label alignment, fold aggregates, reliability."""

from __future__ import annotations

import numpy as np
import pytest

from src.core import EvaluationError
from src.evaluation import runner
from src.evaluation.entities import Scenario
from src.evaluation.metrics import aggregate_metrics


def test_duplicate_folds_are_collapsed(monkeypatch) -> None:
    monkeypatch.setattr(runner, "_available_folds", lambda m, s, k: [0, 1, 2])
    assert runner._resolve_folds("M", Scenario.PMID, 3, [1, 0, 1]) == [0, 1]


def test_fold_out_of_range(monkeypatch) -> None:
    monkeypatch.setattr(runner, "_available_folds", lambda m, s, k: [0, 1])
    with pytest.raises(EvaluationError, match="out of range"):
        runner._resolve_folds("M", Scenario.PMID, 2, [5])


def test_fold_label_sets_must_match() -> None:
    runner._check_labels(["A", "B"], ["A", "B"], fold=1)
    with pytest.raises(EvaluationError, match="fold 1"):
        runner._check_labels(["A", "B"], ["A", "C"], fold=1)


def test_fold_results_use_headline_mask() -> None:
    rng = np.random.default_rng(0)
    Y = [rng.integers(0, 2, size=(30, 3)) for _ in range(2)]
    P = [rng.uniform(size=(30, 3)) for _ in range(2)]
    thr = [np.full((30, 3), 0.5) for _ in range(2)]
    headline = np.array([True, False, True])
    results = runner._fold_results([0, 1], Y, P, thr, headline)
    expected = aggregate_metrics(Y[1], P[1], thr[1], include=headline)["f1_macro"]
    assert results[1].aggregate["f1_macro"] == pytest.approx(expected)


def test_reliability_follows_support_not_headline() -> None:
    Y = np.zeros((40, 2), dtype=int)
    Y[:3, 0] = 1  # 3 positives
    Y[:20, 1] = 1  # 20 positives
    P = np.random.default_rng(0).uniform(size=(40, 2))
    thr = np.full((40, 2), 0.5)
    every = np.array([True, True])
    idx = [np.arange(40)]
    table = runner._class_table(["rare", "common"], every, every, Y, P, thr, idx)
    assert [c.reliable for c in table] == [False, True]
