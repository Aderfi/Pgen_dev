"""Per-class thresholds and temperature fitting on validation."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from src.evaluation.metrics import class_metrics
from src.evaluation.thresholds import fit_temperature, fit_thresholds
from src.model.architectures.config import AxisSpec


def test_threshold_maximizes_f1() -> None:
    y = np.array([[0], [0], [1], [1]])
    p = np.array([[0.1], [0.4], [0.35], [0.8]])
    thr = fit_thresholds(y, p)
    assert class_metrics(y[:, 0], p[:, 0], thr[0])["f1"] == pytest.approx(0.8)


def test_threshold_fallback_without_both_classes() -> None:
    p = np.array([[0.1, 0.9], [0.4, 0.8]])
    thr = fit_thresholds(np.array([[0, 1], [0, 1]]), p)
    assert thr.tolist() == [0.5, 0.5]


def test_temperature_reduces_nll() -> None:
    gen = torch.Generator().manual_seed(0)
    true_logits = torch.randn(2000, 3, generator=gen)
    labels = (torch.rand(2000, 3, generator=gen) < torch.sigmoid(true_logits)).float()
    logits = true_logits * 5  # overconfident
    axes = {
        "phenotype_category": AxisSpec(name="phenotype_category", kind="binary", dim=3)
    }
    scaler = fit_temperature(
        {"phenotype_category": logits}, {"phenotype_category": labels}, axes
    )
    with torch.no_grad():
        scaled = scaler({"phenotype_category": logits})["phenotype_category"]
    before = F.binary_cross_entropy_with_logits(logits, labels)
    after = F.binary_cross_entropy_with_logits(scaled, labels)
    assert after < before
