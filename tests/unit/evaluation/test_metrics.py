"""Per-class, aggregate and calibration metrics."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from src.evaluation.metrics import (
    aggregate_metrics,
    calibration_error,
    class_metrics,
    curves,
)

Y = np.array([1, 0, 1, 1, 0, 0])
P = np.array([0.9, 0.2, 0.6, 0.4, 0.3, 0.7])


def test_class_metrics_match_sklearn() -> None:
    m = class_metrics(Y, P, 0.5)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["specificity"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["f2"] == pytest.approx(2 / 3)
    assert m["f05"] == pytest.approx(2 / 3)
    assert m["roc_auc"] == pytest.approx(roc_auc_score(Y, P))
    assert m["pr_auc"] == pytest.approx(average_precision_score(Y, P))
    assert m["pearson_r"] == pytest.approx(pearsonr(Y, P)[0])
    assert m["pearson_p"] == pytest.approx(pearsonr(Y, P)[1])
    assert m["spearman_rho"] == pytest.approx(spearmanr(Y, P)[0])
    assert m["brier"] == pytest.approx(np.mean((P - Y) ** 2))
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 2)


def test_f2_weights_recall() -> None:
    m = class_metrics(np.array([1, 1, 1, 0]), np.array([0.9, 0.1, 0.1, 0.1]), 0.5)
    # precision 1, recall 1/3 -> F2 = 5 * (1/3) / (4 + 1/3)
    assert m["f2"] == pytest.approx(5 * (1 / 3) / (4 * 1 + 1 / 3))


def test_single_class_column_gives_none() -> None:
    m = class_metrics(np.zeros(6, dtype=int), P, 0.5)
    assert m["roc_auc"] is None
    assert m["pr_auc"] is None
    assert m["pearson_r"] is None
    assert m["mcc"] == 0.0  # sklearn defines MCC as 0 when y is constant


def test_mcc_equals_phi() -> None:
    m = class_metrics(Y, P, 0.5)
    assert m["mcc"] == pytest.approx(pearsonr(Y, (P > 0.5).astype(int))[0])


def test_aggregate_include_mask() -> None:
    Ym = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 1]])
    Pm = np.array([[0.8, 0.3, 0.9], [0.2, 0.7, 0.4], [0.6, 0.4, 0.2], [0.1, 0.2, 0.8]])
    thr = np.full(3, 0.5)
    every = aggregate_metrics(Ym, Pm, thr, include=np.array([True, True, True]))
    some = aggregate_metrics(Ym, Pm, thr, include=np.array([True, False, True]))
    f1s = [class_metrics(Ym[:, c], Pm[:, c], 0.5)["f1"] for c in range(3)]
    assert every["f1_macro"] == pytest.approx(np.mean(f1s))
    assert some["f1_macro"] == pytest.approx(np.mean([f1s[0], f1s[2]]))
    for key in (
        "f1_micro",
        "f1_weighted",
        "f1_samples",
        "f2_macro",
        "f05_micro",
        "precision_macro",
        "recall_samples",
        "roc_auc_macro",
        "roc_auc_micro",
        "pr_auc_macro",
        "pr_auc_micro",
        "subset_accuracy",
        "hamming_loss",
        "jaccard_samples",
        "jaccard_macro",
        "kappa_mean",
        "brier_mean",
        "ece",
        "mce",
    ):
        assert key in every
    assert every["subset_accuracy"] == pytest.approx(0.5)


def test_ece_perfectly_calibrated_is_small() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(size=(20_000, 1))
    y = (rng.uniform(size=p.shape) < p).astype(int)
    ece, mce = calibration_error(y, p)
    assert ece < 0.02
    assert mce >= ece


def test_curves_shapes() -> None:
    c = curves(Y, P)
    assert len(c["pr_precision"]) == len(c["pr_recall"])
    assert len(c["roc_fpr"]) == len(c["roc_tpr"])
    assert sum(c["rel_count"]) == len(Y)


def test_aggregate_matches_sklearn_reference() -> None:
    from sklearn.metrics import (
        cohen_kappa_score,
        fbeta_score,
        hamming_loss,
        jaccard_score,
        precision_score,
    )

    rng = np.random.default_rng(3)
    Ym = rng.integers(0, 2, size=(400, 5))
    Pm = rng.uniform(size=(400, 5))
    thr = rng.uniform(0.3, 0.7, size=5)
    pred = (Pm > thr).astype(int)
    out = aggregate_metrics(Ym, Pm, thr, include=np.ones(5, dtype=bool))
    for avg in ("micro", "macro", "weighted", "samples"):
        for name, beta in (("f1", 1.0), ("f2", 2.0), ("f05", 0.5)):
            ref = fbeta_score(Ym, pred, beta=beta, average=avg, zero_division=0)
            assert out[f"{name}_{avg}"] == pytest.approx(ref)
        ref = precision_score(Ym, pred, average=avg, zero_division=0)
        assert out[f"precision_{avg}"] == pytest.approx(ref)
    assert out["hamming_loss"] == pytest.approx(hamming_loss(Ym, pred))
    assert out["jaccard_samples"] == pytest.approx(
        jaccard_score(Ym, pred, average="samples", zero_division=0)
    )
    assert out["jaccard_macro"] == pytest.approx(
        jaccard_score(Ym, pred, average="macro", zero_division=0)
    )
    aucs = [roc_auc_score(Ym[:, c], Pm[:, c]) for c in range(5)]
    assert out["roc_auc_macro"] == pytest.approx(np.mean(aucs))
    aps = [average_precision_score(Ym[:, c], Pm[:, c]) for c in range(5)]
    assert out["pr_auc_macro"] == pytest.approx(np.mean(aps))
    assert out["roc_auc_micro"] == pytest.approx(roc_auc_score(Ym.ravel(), Pm.ravel()))
    assert out["pr_auc_micro"] == pytest.approx(
        average_precision_score(Ym.ravel(), Pm.ravel())
    )
    kappas = [cohen_kappa_score(Ym[:, c], pred[:, c]) for c in range(5)]
    assert out["kappa_mean"] == pytest.approx(np.mean(kappas))


def test_fast_auc_handles_ties() -> None:
    from src.evaluation.metrics import fast_average_precision, fast_roc_auc

    y = np.array([1, 0, 1, 0, 1, 0, 0])
    p = np.array([0.5, 0.5, 0.2, 0.2, 0.9, 0.1, 0.9])
    assert fast_roc_auc(y, p) == pytest.approx(roc_auc_score(y, p))
    assert fast_average_precision(y, p) == pytest.approx(average_precision_score(y, p))


def test_aggregate_is_fast_enough_for_bootstrap() -> None:
    import time

    rng = np.random.default_rng(0)
    Ym = rng.integers(0, 2, size=(11_000, 7))
    Pm = rng.uniform(size=(11_000, 7))
    thr = np.full(7, 0.5)
    mask = np.ones(7, dtype=bool)
    aggregate_metrics(Ym, Pm, thr, include=mask)
    start = time.perf_counter()
    for _ in range(5):
        aggregate_metrics(Ym, Pm, thr, include=mask)
    assert (time.perf_counter() - start) / 5 < 0.1
