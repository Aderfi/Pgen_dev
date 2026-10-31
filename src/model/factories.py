"""Optimizer + LR-scheduler construction for the two-tower GNN trainer.

A single architecture (``PharmagenTwoTower``) trains here now, so this module
no longer carries the DeepFM-era component registry (``register`` was never
called, and ``SchedulerFactory`` had no callers — both call sites built
``ReduceLROnPlateau`` directly, with inconsistent hardcoded ``factor``/
``patience``). ``optimizer_type`` is the one axis of real variation, and it
is exhaustively enumerated as a ``Literal`` on
:class:`~src.config.models.TrainingParams` instead of a runtime-registered
string key.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from torch import nn

    from src.config.models import TrainingParams


def build_optimizer(
    model: nn.Module,
    params: TrainingParams,
    uncertainty_module: nn.Module | None = None,
) -> torch.optim.Optimizer:
    """Build the optimizer selected by `params.optimizer_type`.

    `uncertainty_module` (the learned per-task log-variances in
    :class:`~src.model.losses.MultiTaskLoss`) gets its own param group with
    no weight decay and an independent learning rate, so L2 regularisation
    doesn't pull the log-variances toward zero.
    """
    param_groups: list[dict[str, Any]] = [
        {
            "params": model.parameters(),
            "weight_decay": params.weight_decay,
            "lr": params.learning_rate,
        }
    ]
    if uncertainty_module is not None:
        param_groups.append(
            {
                "params": uncertainty_module.parameters(),
                "weight_decay": 0.0,
                "lr": params.loss_learning_rate or params.learning_rate,
            }
        )

    if params.optimizer_type == "adamw":
        return torch.optim.AdamW(param_groups)
    if params.optimizer_type == "adam":
        return torch.optim.Adam(param_groups)
    if params.optimizer_type == "sgd":
        return torch.optim.SGD(param_groups, momentum=params.momentum)
    if params.optimizer_type == "rmsprop":
        return torch.optim.RMSprop(param_groups)
    raise ValueError(f"Unknown optimizer_type: {params.optimizer_type!r}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    params: TrainingParams,
) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    """Build the ``ReduceLROnPlateau`` scheduler shared by every trainer."""
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=params.scheduler_factor,
        patience=params.scheduler_patience,
    )


__all__ = ["build_optimizer", "build_scheduler"]
