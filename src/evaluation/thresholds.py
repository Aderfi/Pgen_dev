"""Decision thresholds and probability calibration, fitted on validation only."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import f1_score

from src.model.calibration import TemperatureScaler

if TYPE_CHECKING:
    import torch

    from src.model.architectures.config import AxisSpec

DEFAULT_THRESHOLD = 0.5


def _best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    if not 0 < int(y.sum()) < len(y):
        return DEFAULT_THRESHOLD
    best_t, best_f1 = DEFAULT_THRESHOLD, -1.0
    # Predictions use ``p > t``: candidates just below each observed probability.
    candidates = np.unique(
        np.concatenate([np.nextafter(np.unique(p), -np.inf), [DEFAULT_THRESHOLD]])
    )
    for t in candidates:
        f1 = f1_score(y, (p > t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t


def fit_thresholds(Y_val: np.ndarray, P_val: np.ndarray) -> np.ndarray:
    """Per-class threshold maximizing F1 on validation (0.5 if a class is constant)."""
    Y_val = np.asarray(Y_val).astype(int)
    P_val = np.asarray(P_val, dtype=float)
    return np.array(
        [_best_threshold(Y_val[:, c], P_val[:, c]) for c in range(Y_val.shape[1])]
    )


def fit_temperature(
    logits_val: dict[str, torch.Tensor],
    labels_val: dict[str, torch.Tensor],
    axes: dict[str, AxisSpec],
) -> TemperatureScaler:
    """A :class:`TemperatureScaler` fitted on the validation logits."""
    scaler = TemperatureScaler(axes)
    scaler.fit(logits_val, labels_val, axes)
    return scaler.eval()


__all__ = ["DEFAULT_THRESHOLD", "fit_temperature", "fit_thresholds"]
