"""Component factories for the Pharmagen training pipeline.

Provides :class:`OptimizerFactory`, :class:`SchedulerFactory`, and
:class:`LossFactory` — a registry-based pattern so new components can be
added by registering them rather than editing dispatch logic.
"""

import logging
from collections.abc import Mapping, MutableSequence, Set
from typing import Any

import torch
from torch import nn

from src.model.losses import (
    AdaptiveFocalLoss,
    AsymmetricLoss,
    FocalLoss,
    MultiTaskUncertaintyLoss,
    PolyLoss,
)

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
            param_groups.append({
                "params": uncertainty_module.parameters(),
                "weight_decay": 0.0,
                "lr": params.get("loss_learning_rate", lr),
            })

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


class LossFactory(_ComponentFactory):
    _registry: dict[str, Any] = {
        "cross_entropy": nn.CrossEntropyLoss,
        "bce": nn.BCEWithLogitsLoss,
        "adaptive_focal": AdaptiveFocalLoss,
        "focal": FocalLoss,
        "asymmetric": AsymmetricLoss,
        "poly": PolyLoss,
    }

    @staticmethod
    def create_task_criterions(
        target_cols: MutableSequence[str],
        multi_label_cols: Set[str],
        params: Mapping[str, Any],
        device: torch.device,
    ) -> dict[str, nn.Module]:
        """Return one loss module per target column, moved to *device*."""
        default_ml = params.get("loss_multilabel", "asymmetric")
        default_sl = params.get("loss_singlelabel", "adaptive_focal")

        criterions: dict[str, nn.Module] = {}
        for col in target_cols:
            is_ml = col in multi_label_cols
            loss_key = default_ml if is_ml else default_sl

            kwargs: dict[str, Any] = {}
            if loss_key in {"focal", "adaptive_focal"}:
                kwargs["gamma"] = params.get("gamma", 2.0)
            elif loss_key == "asymmetric":
                kwargs["gamma_neg"] = params.get("gamma_neg", 4.0)
                kwargs["gamma_pos"] = params.get("gamma_pos", 1.0)
                kwargs["clip"] = params.get("asl_clip", 0.05)

            loss_cls = LossFactory.get(loss_key) or nn.BCEWithLogitsLoss
            criterions[col] = loss_cls(**kwargs).to(device)

        return criterions

    @staticmethod
    def create_uncertainty_wrapper(
        tasks: list[str], device: torch.device
    ) -> MultiTaskUncertaintyLoss:
        return MultiTaskUncertaintyLoss(tasks=tasks).to(device)
