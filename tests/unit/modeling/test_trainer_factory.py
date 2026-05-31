"""Tests for trainer subclass dispatch and TrainingLoop input validation."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import src.model.factories as module_builder_mod
from src.model.training import (
    OptunaTrialTrainer,
    StandardTrainer,
    TrainingLoop,
)

# ---------------------------------------------------------------------- fixtures


class _StubLossFactory:
    """The real LossFactory pulls in losses + multi-task uncertainty wiring;
    for unit tests we just need any criterion that maps target names to a
    nn.Module."""

    @staticmethod
    def create_task_criterions(
        target_cols: list[str],
        multi_label_cols: set[str],  # noqa: ARG004
        params: dict,  # noqa: ARG004
        device: torch.device,  # noqa: ARG004
    ) -> dict[str, nn.Module]:
        return {col: nn.CrossEntropyLoss() for col in target_cols}


@pytest.fixture(autouse=True)
def stub_loss_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module_builder_mod, "LossFactory", _StubLossFactory)


@pytest.fixture
def trainer_kwargs() -> dict[str, Any]:
    model = nn.Linear(10, 5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
    return {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "device": torch.device("cpu"),
        "target_cols": ["cls"],
        "multi_label_cols": set(),
        "params": {"learning_rate": 1e-3, "weight_decay": 1e-4},
    }


# -------------------------------------------------------------- input validation


class TestInputValidation:
    def test_empty_target_cols_rejected(self, trainer_kwargs: dict[str, Any]) -> None:
        trainer_kwargs["target_cols"] = []
        with pytest.raises(ValueError, match="target_cols"):
            StandardTrainer(**trainer_kwargs)

    def test_non_set_multi_label_cols_rejected(
        self, trainer_kwargs: dict[str, Any]
    ) -> None:
        trainer_kwargs["multi_label_cols"] = ["cls"]  # list, not set
        with pytest.raises(TypeError, match="multi_label_cols"):
            StandardTrainer(**trainer_kwargs)

    def test_non_torch_device_rejected(self, trainer_kwargs: dict[str, Any]) -> None:
        trainer_kwargs["device"] = "cpu"  # str, not torch.device
        with pytest.raises(TypeError, match="device"):
            StandardTrainer(**trainer_kwargs)

    def test_non_mapping_params_rejected(self, trainer_kwargs: dict[str, Any]) -> None:
        trainer_kwargs["params"] = ["learning_rate", 1e-3]  # list, not Mapping
        with pytest.raises(TypeError, match="params"):
            StandardTrainer(**trainer_kwargs)


# -------------------------------------------------------- common interface


class TestSharedInterface:
    def test_standard_has_checkpoint_manager(
        self, trainer_kwargs: dict[str, Any]
    ) -> None:
        trainer = StandardTrainer(**trainer_kwargs)
        assert hasattr(trainer, "checkpoint_manager")
        assert trainer.checkpoint_manager is not None

    def test_optuna_has_no_checkpoint_manager(
        self, trainer_kwargs: dict[str, Any]
    ) -> None:
        trainer = OptunaTrialTrainer(**trainer_kwargs)
        # Critical contract: the optuna trainer must NOT carry a checkpoint
        # manager — the tuner persists best trials separately.
        assert not hasattr(trainer, "checkpoint_manager")

    def test_both_share_loss_fns(self, trainer_kwargs: dict[str, Any]) -> None:
        a = StandardTrainer(**trainer_kwargs)
        b = OptunaTrialTrainer(**trainer_kwargs)
        assert "cls" in a.loss_fns
        assert "cls" in b.loss_fns

    def test_both_track_best_loss(self, trainer_kwargs: dict[str, Any]) -> None:
        for cls in (StandardTrainer, OptunaTrialTrainer):
            t = cls(**trainer_kwargs)
            assert t.best_loss == float("inf")
            assert t.patience_counter == 0
            assert t.current_epoch == 0


class TestNanCheck:
    def test_nan_raises_training_error_in_base(self) -> None:
        from src.core import TrainingError

        with pytest.raises(TrainingError, match="NaN"):
            TrainingLoop._check_nan(float("nan"), epoch=5)

    def test_finite_passes(self) -> None:
        # Should not raise for finite values.
        TrainingLoop._check_nan(0.0, epoch=1)
        TrainingLoop._check_nan(1e10, epoch=1)
