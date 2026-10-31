"""Classification metrics for multi-label outputs.

All functions take numpy arrays: ``Y`` ``(n, C)`` in ``{0, 1}``, ``P`` ``(n, C)``
probabilities and ``thr`` ``(C,)`` per-class thresholds. A metric that is
undefined on the given data (e.g. ROC-AUC with a single class present) is
``None``, never NaN, so it serializes cleanly and the report can say so.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
from scipy.stats import pearsonr, rankdata, spearmanr
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    cohen_kappa_score,
    fbeta_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

N_CALIBRATION_BINS = 10
CLASS_METRIC_NAMES = (
    "precision",
    "recall",
    "specificity",
    "f1",
    "f2",
    "f05",
    "roc_auc",
    "pr_auc",
    "mcc",
    "brier",
    "kappa",
    "pearson_r",
    "pearson_p",
    "spearman_rho",
    "spearman_p",
)
_AVERAGES = ("micro", "macro", "weighted", "samples")
_BETAS = {"f1": 1.0, "f2": 2.0, "f05": 0.5}


def _finite(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _both_classes(y: np.ndarray) -> bool:
    return 0 < int(y.sum()) < len(y)


def _correlation(fn, y: np.ndarray, p: np.ndarray) -> tuple[float | None, float | None]:
    if not _both_classes(y) or np.ptp(p) == 0:
        return None, None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, pvalue = fn(y, p)
    return _finite(stat), _finite(pvalue)


def class_metrics(
    y: np.ndarray, p: np.ndarray, threshold: float | np.ndarray
) -> dict[str, float | None]:
    """Every per-class metric of one label column at ``threshold`` (scalar or per row)."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p > threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    out: dict[str, float | None] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    out["precision"] = tp / (tp + fp) if tp + fp else 0.0
    out["recall"] = tp / (tp + fn) if tp + fn else 0.0
    out["specificity"] = tn / (tn + fp) if tn + fp else 0.0
    for name, beta in _BETAS.items():
        out[name] = float(fbeta_score(y, pred, beta=beta, zero_division=0))
    both = _both_classes(y)
    out["roc_auc"] = _finite(roc_auc_score(y, p)) if both else None
    out["pr_auc"] = _finite(average_precision_score(y, p)) if both else None
    out["mcc"] = float(matthews_corrcoef(y, pred))
    out["brier"] = float(brier_score_loss(y, p, pos_label=1)) if len(y) else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out["kappa"] = (
            _finite(cohen_kappa_score(y, pred)) if len(set(y) | set(pred)) > 1 else None
        )
    out["pearson_r"], out["pearson_p"] = _correlation(pearsonr, y, p)
    out["spearman_rho"], out["spearman_p"] = _correlation(spearmanr, y, p)
    return out


def calibration_error(
    Y: np.ndarray, P: np.ndarray, n_bins: int = N_CALIBRATION_BINS
) -> tuple[float, float]:
    """(ECE, MCE) over all (row, class) cells, equal-width bins."""
    y = np.asarray(Y, dtype=float).ravel()
    p = np.asarray(P, dtype=float).ravel()
    if not len(p):
        return 0.0, 0.0
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece, mce = 0.0, 0.0
    for b in range(n_bins):
        mask = bins == b
        if not mask.any():
            continue
        gap = abs(p[mask].mean() - y[mask].mean())
        ece += mask.mean() * gap
        mce = max(mce, gap)
    return float(ece), float(mce)


def _mean_defined(values: list[float | None]) -> float | None:
    defined = [v for v in values if v is not None]
    return float(np.mean(defined)) if defined else None


def _fbeta(tp, fp, fn, beta: float):
    b2 = beta * beta
    denom = (1 + b2) * tp + b2 * fn + fp
    return np.divide(
        (1 + b2) * tp, denom, out=np.zeros_like(denom, dtype=float), where=denom > 0
    )


def _ratio(num, den):
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    return np.divide(num, den, out=np.zeros_like(den), where=den > 0)


def _averaged(per_class, micro, support, per_row) -> dict[str, float]:
    total = support.sum()
    return {
        "micro": float(micro),
        "macro": float(per_class.mean()),
        "weighted": float((per_class * support).sum() / total) if total else 0.0,
        "samples": float(per_row.mean()),
    }


def fast_roc_auc(y: np.ndarray, p: np.ndarray) -> float | None:
    """ROC-AUC via the Mann-Whitney rank statistic (equals sklearn, ties averaged)."""
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if not n_pos or not n_neg:
        return None
    ranks = rankdata(p)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def fast_average_precision(y: np.ndarray, p: np.ndarray) -> float | None:
    """Average precision over distinct thresholds (equals sklearn's)."""
    if not _both_classes(y):
        return None
    order = np.argsort(-p, kind="mergesort")
    ys, ps = y[order], p[order]
    last = np.r_[np.flatnonzero(np.diff(ps)), len(ps) - 1]
    tps = np.cumsum(ys)[last]
    precision = tps / (last + 1)
    recall = tps / tps[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def _kappa(tp, fp, fn, tn) -> np.ndarray:
    n = tp + fp + fn + tn
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)
    return np.where(pe < 1, (po - pe) / np.where(pe < 1, 1 - pe, 1), np.nan)


def aggregate_metrics(
    Y: np.ndarray, P: np.ndarray, thr: np.ndarray, *, include: np.ndarray
) -> dict[str, float | None]:
    """Averaged metrics over the classes selected by the boolean mask ``include``.

    ``thr`` is per class ``(C,)`` or per row and class ``(n, C)`` (out-of-fold
    predictions keep each fold's own thresholds). Computed from confusion
    counts with numpy (sklearn-equivalent, ``zero_division=0``) because it runs
    once per bootstrap resample.
    """
    Y = np.asarray(Y).astype(bool)[:, include]
    P = np.asarray(P, dtype=float)[:, include]
    out: dict[str, float | None] = {}
    if Y.shape[1] == 0 or Y.shape[0] == 0:
        return out
    pred = P > np.asarray(thr)[..., include]  # thr: (C,) or (n, C)
    hit, miss = pred & Y, pred != Y
    tp, fp, fn = hit.sum(0), (pred & ~Y).sum(0), (~pred & Y).sum(0)
    tn = len(Y) - tp - fp - fn
    rtp, rfp, rfn = hit.sum(1), (pred & ~Y).sum(1), (~pred & Y).sum(1)
    support = tp + fn
    stp, sfp, sfn = tp.sum(), fp.sum(), fn.sum()
    for name, beta in _BETAS.items():
        avg = _averaged(
            _fbeta(tp, fp, fn, beta),
            _fbeta(stp, sfp, sfn, beta),
            support,
            _fbeta(rtp, rfp, rfn, beta),
        )
        out |= {f"{name}_{k}": v for k, v in avg.items()}
    for name, den, rden, sden in (
        ("precision", tp + fp, rtp + rfp, stp + sfp),
        ("recall", tp + fn, rtp + rfn, stp + sfn),
    ):
        avg = _averaged(_ratio(tp, den), _ratio(stp, sden), support, _ratio(rtp, rden))
        out |= {f"{name}_{k}": v for k, v in avg.items()}
    out["subset_accuracy"] = float((~miss).all(axis=1).mean())
    out["hamming_loss"] = float(miss.mean())
    out["jaccard_samples"] = float(_ratio(rtp, rtp + rfp + rfn).mean())
    out["jaccard_macro"] = float(_ratio(tp, tp + fp + fn).mean())
    kappa = _kappa(tp, fp, fn, tn)
    out["kappa_mean"] = float(np.nanmean(kappa)) if np.isfinite(kappa).any() else None
    Yi = Y.astype(int)
    aucs = [fast_roc_auc(Yi[:, c], P[:, c]) for c in range(Y.shape[1])]
    aps = [fast_average_precision(Yi[:, c], P[:, c]) for c in range(Y.shape[1])]
    out["roc_auc_macro"] = _mean_defined(aucs)
    out["pr_auc_macro"] = _mean_defined(aps)
    out["roc_auc_micro"] = fast_roc_auc(Yi.ravel(), P.ravel())
    out["pr_auc_micro"] = fast_average_precision(Yi.ravel(), P.ravel())
    out["brier_mean"] = float(np.mean((P - Yi) ** 2))
    out["ece"], out["mce"] = calibration_error(Yi, P)
    return out


def curves(
    y: np.ndarray, p: np.ndarray, n_bins: int = N_CALIBRATION_BINS
) -> dict[str, list[float]]:
    """PR, ROC and reliability-diagram points of one label column."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    out: dict[str, list[float]] = {}
    if _both_classes(y):
        prec, rec, _ = precision_recall_curve(y, p)
        fpr, tpr, _ = roc_curve(y, p)
        out |= {
            "pr_precision": prec.tolist(),
            "pr_recall": rec.tolist(),
            "roc_fpr": fpr.tolist(),
            "roc_tpr": tpr.tolist(),
        }
    else:
        out |= {"pr_precision": [], "pr_recall": [], "roc_fpr": [], "roc_tpr": []}
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    mean_pred, frac_pos, count = [], [], []
    for b in range(n_bins):
        mask = bins == b
        count.append(int(mask.sum()))
        mean_pred.append(float(p[mask].mean()) if mask.any() else (b + 0.5) / n_bins)
        frac_pos.append(float(y[mask].mean()) if mask.any() else 0.0)
    out |= {"rel_mean_pred": mean_pred, "rel_frac_pos": frac_pos, "rel_count": count}
    return out


__all__ = [
    "CLASS_METRIC_NAMES",
    "aggregate_metrics",
    "calibration_error",
    "class_metrics",
    "curves",
]
