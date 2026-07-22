"""Tests for trainer subclass dispatch, TrainingLoop input validation, and
MultiTaskLoss / CompositionalLabelLoss wiring into the training step."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

from src.model.architectures.config import AxisSpec
from src.model.architectures.heads.axis_heads import AxisHeads
from src.model.architectures.heads.compose import ComposeHead
from src.model.losses import CompositionalLabelLoss, MultiTaskLoss
from src.model.training import (
    OptunaTrialTrainer,
    StandardTrainer,
    TrainingLoop,
)

# ---------------------------------------------------------------------- fixtures


@pytest.fixture
def axes() -> dict[str, AxisSpec]:
    return {"cls": AxisSpec(name="cls", dim=3, kind="multiclass", embedding_dim=8)}


@pytest.fixture
def trainer_kwargs(axes: dict[str, AxisSpec]) -> dict[str, Any]:
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
        "multitask_loss": MultiTaskLoss(axes),
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

    def test_both_share_multitask_loss(self, trainer_kwargs: dict[str, Any]) -> None:
        a = StandardTrainer(**trainer_kwargs)
        b = OptunaTrialTrainer(**trainer_kwargs)
        assert "cls" in a.multitask_loss.specs
        assert "cls" in b.multitask_loss.specs

    def test_both_track_best_loss(self, trainer_kwargs: dict[str, Any]) -> None:
        for cls in (StandardTrainer, OptunaTrialTrainer):
            t = cls(**trainer_kwargs)
            assert t.best_loss == float("inf")
            assert t.patience_counter == 0
            assert t.current_epoch == 0

    def test_raw_model_reference_kept(self, trainer_kwargs: dict[str, Any]) -> None:
        # StandardTrainer compiles `self.model`; `_raw_model` must stay the
        # original uncompiled module for structural attribute access.
        trainer = StandardTrainer(**trainer_kwargs)
        assert trainer._raw_model is trainer_kwargs["model"]


class TestNanCheck:
    def test_nan_raises_training_error_in_base(self) -> None:
        from src.core import TrainingError

        with pytest.raises(TrainingError, match="NaN"):
            TrainingLoop._check_nan(float("nan"), epoch=5)

    def test_finite_passes(self) -> None:
        # Should not raise for finite values.
        TrainingLoop._check_nan(0.0, epoch=1)
        TrainingLoop._check_nan(1e10, epoch=1)


# ------------------------------------------------------------ loss wiring


class _FakeGNN(nn.Module):
    """Minimal stand-in for `PharmagenTwoTower`'s structural surface.

    Exposes real `AxisHeads` / `ComposeHead` submodules (so
    `single_label_axes()` / `embed_tuples()` behave exactly like the real
    model) but a trivial forward pass that ignores its inputs' graph
    structure and treats `drug_data` as the shared representation.
    """

    def __init__(
        self, axes: dict[str, AxisSpec], out_dim: int = 4, *, compose: bool = True
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4)
        self.axis_heads = AxisHeads(in_dim=4, axes=axes)
        self.compose = ComposeHead(axes=axes, out_dim=out_dim) if compose else None

    def forward(self, drug_data: torch.Tensor, geno_data: torch.Tensor) -> dict:
        z = self.proj(drug_data) + geno_data.sum() * 0.0
        logits = self.axis_heads(z)
        outputs: dict[str, torch.Tensor] = dict(logits)
        if self.compose is not None:
            outputs["_z"] = self.compose(logits, self.axis_heads.axis_embeddings)
        return outputs


def _make_batch(batch_size: int = 6) -> dict[str, Any]:
    return {
        "drug_batch": torch.randn(batch_size, 4),
        "geno_batch": torch.randn(batch_size, 1),
        "targets": {"cls": torch.randint(0, 3, (batch_size,))},
    }


class TestMultiTaskLossWiring:
    def test_compute_step_produces_finite_loss(self, axes: dict[str, AxisSpec]) -> None:
        model = _FakeGNN(axes, compose=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
        multitask_loss = MultiTaskLoss(axes)

        trainer = OptunaTrialTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=torch.device("cpu"),
            target_cols=["cls"],
            multi_label_cols=set(),
            params={},
            multitask_loss=multitask_loss,
        )

        loss, metrics = trainer._compute_step(_make_batch())
        assert torch.isfinite(loss)
        assert loss.requires_grad
        loss.backward()
        assert "loss" in metrics and "acc" in metrics

    def test_compose_loss_adds_to_total_when_z_present(
        self, axes: dict[str, AxisSpec]
    ) -> None:
        model = _FakeGNN(axes, compose=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
        multitask_loss = MultiTaskLoss(axes)
        compose_loss = CompositionalLabelLoss()

        trainer = OptunaTrialTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=torch.device("cpu"),
            target_cols=["cls"],
            multi_label_cols=set(),
            params={},
            multitask_loss=multitask_loss,
            compose_loss=compose_loss,
            compose_weight=0.5,
        )

        batch = _make_batch()
        outputs = trainer.model(
            batch["drug_batch"].to(trainer.device),
            batch["geno_batch"].to(trainer.device),
        )
        targets = {k: v.to(trainer.device) for k, v in batch["targets"].items()}
        assert "_z" in outputs

        total_with_compose, _ = trainer._calculate_loss_and_metrics(outputs, targets)

        # Disabling the compose term must strictly drop the total loss by the
        # (non-negative, generically non-zero) compose contribution.
        trainer.compose_loss = None
        total_without_compose, _ = trainer._calculate_loss_and_metrics(outputs, targets)

        assert torch.isfinite(total_with_compose)
        assert not torch.allclose(total_with_compose, total_without_compose)

    def test_no_compose_term_when_model_has_no_compose_head(
        self, axes: dict[str, AxisSpec]
    ) -> None:
        model = _FakeGNN(axes, compose=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
        multitask_loss = MultiTaskLoss(axes)
        compose_loss = CompositionalLabelLoss()

        trainer = OptunaTrialTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=torch.device("cpu"),
            target_cols=["cls"],
            multi_label_cols=set(),
            params={},
            multitask_loss=multitask_loss,
            compose_loss=compose_loss,
        )

        batch = _make_batch()
        outputs = trainer.model(batch["drug_batch"], batch["geno_batch"])
        assert "_z" not in outputs

        total, _ = trainer._calculate_loss_and_metrics(outputs, batch["targets"])
        expected_total, _ = multitask_loss(outputs, batch["targets"])
        assert torch.allclose(total, expected_total)
