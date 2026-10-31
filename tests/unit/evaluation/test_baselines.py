"""Trivial baselines the model must beat."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.baselines import key_frequency_baseline, prevalence_baseline


def test_prevalence_baseline() -> None:
    y = np.array([[1, 0], [0, 0], [1, 1], [1, 0]])
    out = prevalence_baseline(y, 3)
    assert out.shape == (3, 2)
    assert out[0].tolist() == pytest.approx([0.75, 0.25])


def test_key_frequency_uses_train_only() -> None:
    train_keys = [frozenset({"a"})] * 4 + [frozenset({"b"})] * 4
    y = np.array([[1], [1], [1], [0], [0], [0], [0], [1]])
    out = key_frequency_baseline(train_keys, y, [frozenset({"a"}), frozenset({"z"})])
    assert out[0, 0] == pytest.approx(0.75)
    assert out[1, 0] == pytest.approx(y.mean())


def test_multi_key_rows_average() -> None:
    train_keys = [frozenset({"a"})] * 4 + [frozenset({"b"})] * 4
    y = np.array([[1], [1], [1], [0], [0], [0], [0], [1]])
    out = key_frequency_baseline(train_keys, y, [frozenset({"a", "b"})])
    assert out[0, 0] == pytest.approx(0.5)
