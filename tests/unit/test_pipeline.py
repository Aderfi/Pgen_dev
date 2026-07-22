"""Unit tests for ``src.pipeline`` helpers that don't need a real dataset.

``_build_label_table`` only reads ``.targets`` (dict of axis -> tensor of
class indices) and ``.target_encoder.encoders`` (dict of axis -> an object
exposing ``inverse_transform``) off its ``train_dataset`` argument, so a
lightweight stub stands in for the real ``DoubleTowerDataset``.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from src.pipeline import _build_label_table


class _FakeEncoder:
    """Minimal stand-in for a fitted ``sklearn.LabelEncoder``."""

    def __init__(self, classes: list[str]) -> None:
        self.classes = classes

    def inverse_transform(self, indices: list[int]) -> list[str]:
        return [self.classes[i] for i in indices]


def _make_train_dataset(
    targets: dict[str, torch.Tensor], encoders: dict
) -> SimpleNamespace:
    return SimpleNamespace(
        targets=targets,
        target_encoder=SimpleNamespace(encoders=encoders),
    )


def test_build_label_table_unique_tuples_and_labels():
    targets = {
        "dir": torch.tensor([0, 1, 0, 1]),
        "pheno": torch.tensor([0, 2, 0, 1]),
    }
    encoders = {
        "dir": _FakeEncoder(["increase", "decrease"]),
        "pheno": _FakeEncoder(["poor", "normal", "rapid"]),
    }
    train_dataset = _make_train_dataset(targets, encoders)

    tuples, labels = _build_label_table(train_dataset, ["dir", "pheno"])

    # Row 2 duplicates row 0's tuple: only 3 unique tuples over 4 rows.
    assert tuples == [(0, 0), (1, 2), (1, 1)]
    assert labels == [
        "dir=increase|pheno=poor",
        "dir=decrease|pheno=rapid",
        "dir=decrease|pheno=normal",
    ]


def test_build_label_table_respects_composable_column_order():
    targets = {
        "dir": torch.tensor([1]),
        "pheno": torch.tensor([2]),
    }
    encoders = {
        "dir": _FakeEncoder(["increase", "decrease"]),
        "pheno": _FakeEncoder(["poor", "normal", "rapid"]),
    }
    train_dataset = _make_train_dataset(targets, encoders)

    tuples, labels = _build_label_table(train_dataset, ["pheno", "dir"])

    assert tuples == [(2, 1)]
    assert labels == ["pheno=rapid|dir=decrease"]


def test_build_label_table_empty_composable_returns_empty_lists():
    train_dataset = _make_train_dataset({}, {})

    tuples, labels = _build_label_table(train_dataset, [])

    assert tuples == []
    assert labels == []
