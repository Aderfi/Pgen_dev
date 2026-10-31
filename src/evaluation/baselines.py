"""Trivial baselines: what the model has to beat to be using the genotype.

* ``prevalence`` — every row gets the train label frequencies;
* ``drug_only`` — the label frequencies of the row's drug(s) in train;
* ``gene_only`` — the same per gene.

If the model does not beat ``drug_only`` significantly, it is not using the
genotype tower.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

BASELINES = ("prevalence", "drug_only", "gene_only")


def prevalence_baseline(Y_train: np.ndarray, n: int) -> np.ndarray:
    """``(n, C)`` rows equal to the train label frequencies."""
    prevalence = np.asarray(Y_train, dtype=float).mean(axis=0)
    return np.tile(prevalence, (n, 1))


def key_frequency_baseline(
    train_keys: list[frozenset[str]],
    Y_train: np.ndarray,
    test_keys: list[frozenset[str]],
) -> np.ndarray:
    """Mean over a row's keys of their train label frequency (prevalence if unseen)."""
    Y_train = np.asarray(Y_train, dtype=float)
    sums: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(Y_train.shape[1]))
    counts: dict[str, int] = defaultdict(int)
    for keys, y in zip(train_keys, Y_train, strict=True):
        for key in keys:
            sums[key] += y
            counts[key] += 1
    prevalence = Y_train.mean(axis=0)
    out = np.empty((len(test_keys), Y_train.shape[1]))
    for i, keys in enumerate(test_keys):
        seen = [sums[k] / counts[k] for k in keys if counts.get(k)]
        out[i] = np.mean(seen, axis=0) if seen else prevalence
    return out


__all__ = ["BASELINES", "key_frequency_baseline", "prevalence_baseline"]
