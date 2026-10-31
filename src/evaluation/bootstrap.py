"""Bootstrap confidence intervals and paired bootstrap tests.

The resample indices are drawn once per evaluation and reused for every
metric (and for the model and each baseline), so intervals and paired
differences are computed on exactly the same resamples. With ``groups`` the
unit of resampling is the group (cluster bootstrap): rows that share an
entity are not independent, and resampling them one by one would make the
intervals too narrow.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from src.evaluation.schemas import MetricCI

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

MIN_GROUPS = 20


def resample_indices(
    n: int, groups: np.ndarray | None, n_boot: int, seed: int
) -> list[np.ndarray]:
    """``n_boot`` index arrays: rows, or whole groups, drawn with replacement."""
    rng = np.random.default_rng(seed)
    if groups is None:
        return [rng.integers(0, n, size=n) for _ in range(n_boot)]
    groups = np.asarray(groups)
    uniques, inverse = np.unique(groups, return_inverse=True)
    if len(uniques) < MIN_GROUPS:
        logger.warning(
            "Cluster bootstrap over only %d groups; intervals will be unstable",
            len(uniques),
        )
    members = [np.flatnonzero(inverse == g) for g in range(len(uniques))]
    out = []
    for _ in range(n_boot):
        picked = rng.integers(0, len(uniques), size=len(uniques))
        out.append(np.concatenate([members[g] for g in picked]))
    return out


def _values(
    fn: Callable[[np.ndarray], float | None], indices: list[np.ndarray]
) -> np.ndarray:
    vals = [fn(i) for i in indices]
    return np.array([v for v in vals if v is not None], dtype=float)


def bootstrap_ci(
    fn: Callable[[np.ndarray], float | None],
    indices: list[np.ndarray],
    *,
    point: float | None,
    alpha: float = 0.05,
) -> MetricCI:
    """Percentile ``1 - alpha`` interval of ``fn`` over the resamples."""
    vals = _values(fn, indices)
    if not len(vals) or point is None:
        return MetricCI(value=point)
    low, high = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return MetricCI(value=point, low=float(low), high=float(high))


def paired_test(
    fn_a: Callable[[np.ndarray], float | None],
    fn_b: Callable[[np.ndarray], float | None],
    indices: list[np.ndarray],
    *,
    point_a: float | None,
    point_b: float | None,
    alpha: float = 0.05,
) -> tuple[MetricCI, float | None]:
    """CI of ``a - b`` and its two-sided bootstrap p-value (H0: no difference)."""
    if point_a is None or point_b is None:
        return MetricCI(value=None), None
    diffs = []
    for i in indices:
        a, b = fn_a(i), fn_b(i)
        if a is not None and b is not None:
            diffs.append(a - b)
    point = point_a - point_b
    if not diffs:
        return MetricCI(value=point), None
    d = np.array(diffs)
    low, high = np.quantile(d, [alpha / 2, 1 - alpha / 2])
    p = 2 * min(float((d <= 0).mean()), float((d >= 0).mean()))
    p = min(max(p, 1 / (len(d) + 1)), 1.0)
    return MetricCI(value=point, low=float(low), high=float(high)), p


__all__ = ["MIN_GROUPS", "bootstrap_ci", "paired_test", "resample_indices"]
