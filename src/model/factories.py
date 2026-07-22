"""Component factories for the Pharmagen training pipeline.

Provides :class:`OptimizerFactory` and :class:`SchedulerFactory` — a
registry-based pattern so new components can be added by registering them
rather than editing dispatch logic.
"""

import logging
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)


class _ComponentFactory:
    """Registry base: maps string keys to constructors."""

    _registry: dict[str, Any] = {}

    @classmethod
    def register(cls, name: str, component: Any) -> None:
        cls._registry[name] = component

    @classmethod
    def get(cls, name: str) -> Any:
        return cls._registry.get(name)


class OptimizerFactory(_ComponentFactory):
    _registry: dict[str, Any] = {
        "adamw": torch.optim.AdamW,
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
    }

    @staticmethod
    def create(
        model: nn.Module,
        params: dict[str, Any],
        uncertainty_module: nn.Module | None = None,
    ) -> torch.optim.Optimizer:
        lr = params.get("learning_rate", 1e-3)
        wd = params.get("weight_decay", 1e-4)
        opt_name = params.get("optimizer_type", "adamw").lower()

        param_groups: list[dict[str, Any]] = [
            {"params": model.parameters(), "weight_decay": wd, "lr": lr}
        ]
        if uncertainty_module is not None:
            param_groups.append(
                {
                    "params": uncertainty_module.parameters(),
                    "weight_decay": 0.0,
                    "lr": params.get("loss_learning_rate", lr),
                }
            )

        optimizer_cls = OptimizerFactory.get(opt_name) or torch.optim.AdamW
        kwargs: dict[str, Any] = {}
        if opt_name == "sgd":
            kwargs["momentum"] = params.get("momentum", 0.9)

        return optimizer_cls(param_groups, **kwargs)


class SchedulerFactory(_ComponentFactory):
    _registry: dict[str, Any] = {
        "plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
    }

    @staticmethod
    def create(
        optimizer: torch.optim.Optimizer,
        params: dict[str, Any],
    ) -> torch.optim.lr_scheduler.LRScheduler | None:
        stype = params.get("scheduler_type", "plateau").lower()
        if stype == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=params.get("scheduler_factor", 0.5),
                patience=params.get("scheduler_patience", 3),
            )
        if stype == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=params.get("epochs", 50), eta_min=1e-6
            )
        return None
