"""Bootstrap confidence intervals and paired tests."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from src.evaluation.bootstrap import bootstrap_ci, paired_test, resample_indices


def test_ci_contains_point_and_is_ordered() -> None:
    x = np.random.default_rng(0).normal(size=500)
    idx = resample_indices(500, None, 2000, seed=1)
    ci = bootstrap_ci(lambda i: float(x[i].mean()), idx, point=float(x.mean()))
    assert ci.low < ci.value < ci.high
    assert (ci.high - ci.low) == pytest.approx(2 * 1.96 / np.sqrt(500), rel=0.3)


def test_cluster_resampling_keeps_groups_whole() -> None:
    groups = np.array([0, 0, 1, 1, 2, 2])
    for sample in resample_indices(6, groups, 50, seed=0):
        assert len(sample) % 2 == 0
        pairs = sample.reshape(-1, 2)
        assert (groups[pairs[:, 0]] == groups[pairs[:, 1]]).all()


def test_all_none_gives_point_only() -> None:
    idx = resample_indices(10, None, 20, seed=0)
    ci = bootstrap_ci(lambda i: None, idx, point=0.3)
    assert (ci.value, ci.low, ci.high) == (0.3, None, None)


def test_paired_identical_models_p_is_one() -> None:
    x = np.random.default_rng(0).uniform(size=200)
    idx = resample_indices(200, None, 500, seed=0)
    fn = lambda i: float(x[i].mean())  # noqa: E731
    diff, p = paired_test(fn, fn, idx, point_a=float(x.mean()), point_b=float(x.mean()))
    assert p == 1.0
    assert (diff.low, diff.high) == (0.0, 0.0)


def test_paired_clear_difference_small_p() -> None:
    x = np.random.default_rng(0).uniform(size=200)
    idx = resample_indices(200, None, 500, seed=0)
    diff, p = paired_test(
        lambda i: float(x[i].mean()) + 0.2,
        lambda i: float(x[i].mean()),
        idx,
        point_a=float(x.mean()) + 0.2,
        point_b=float(x.mean()),
    )
    assert p == pytest.approx(1 / 501)
    assert diff.value == pytest.approx(0.2)


def test_few_groups_warns(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        resample_indices(10, np.arange(10) % 5, 10, seed=0)
    assert any("groups" in r.message for r in caplog.records)
