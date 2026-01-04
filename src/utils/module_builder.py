# Pharmagen - Pharmacogenetic Prediction and Therapeutic Efficacy
# Copyright (C) 2025 Adrim Hamed Outmani
# src/utils/factory.py
import logging
from typing import Any, Dict, List, Optional, Set

import torch
import torch.nn as nn

from src.utils.losses import (
    AdaptiveFocalLoss,
    AsymmetricLoss,
    FocalLoss,
    MultiTaskUncertaintyLoss,
    PolyLoss,
)

logger = logging.getLogger(__name__)


class ComponentFactory:
    """
    Base Factory class implementing a Registry pattern.
    Adheres to OCP: New components can be registered without modifying the factory logic.
    """

    _registry: dict[str, Any] = {}

    @classmethod
    def register(cls, name: str, component: Any) -> None:
        cls._registry[name] = component

    @classmethod
    def get(cls, name: str) -> Any:
        return cls._registry.get(name)


class OptimizerFactory(ComponentFactory):
    _registry = {
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

        # Parameter Groups construction
        param_groups = [{"params": model.parameters(), "weight_decay": wd, "lr": lr}]

        # Uncertainty module handling (Special case handled cleanly)
        if uncertainty_module:
            param_groups.append(
                {
                    "params": uncertainty_module.parameters(),
                    "weight_decay": 0.0,
                    "lr": params.get("loss_learning_rate", lr),
                }
            )

        optimizer_cls = OptimizerFactory.get(opt_name) or torch.optim.Adam

        kwargs = {}
        if opt_name == "sgd":
            kwargs["momentum"] = params.get("momentum", 0.9)

        return optimizer_cls(param_groups, **kwargs)


class SchedulerFactory(ComponentFactory):
    _registry = {
        "plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
    }

    @staticmethod
    def create(
        optimizer: torch.optim.Optimizer, params: dict[str, Any]
    ) -> torch.optim.lr_scheduler.LRScheduler | None:
        stype = params.get("scheduler_type", "plateau").lower()

        if stype == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=params.get("scheduler_factor", 0.5),
                patience=params.get("scheduler_patience", 3),
            )
        elif stype == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=params.get("epochs", 50), eta_min=1e-6
            )
        return None


class LossFactory(ComponentFactory):
    _registry = {
        "cross_entropy": nn.CrossEntropyLoss,
        "bce": nn.BCEWithLogitsLoss,
        "adaptive_focal": AdaptiveFocalLoss,
        "focal": FocalLoss,
        "asymmetric": AsymmetricLoss,
        "poly": PolyLoss,
    }

    @staticmethod
    def create_task_criterions(
        target_cols: list[str],
        multi_label_cols: set[str],
        params: dict[str, Any],
        device: torch.device,
    ) -> dict[str, nn.Module]:
        """
        Orquesta la creación de múltiples funciones de pérdida basadas en la configuración.
        """
        criterions = {}
        # Recuperamos las preferencias globales definidas en la config
        default_ml = params.get("loss_multilabel", "asymmetric")
        default_sl = params.get("loss_singlelabel", "adaptive_focal")

        for col in target_cols:
            is_ml = col in multi_label_cols
            loss_key = default_ml if is_ml else default_sl

            # Construcción dinámica de argumentos para la pérdida
            kwargs = {}
            if loss_key in ["focal", "adaptive_focal"]:
                kwargs["gamma"] = params.get("gamma", 2.0)  # Conectado a modelos.toml

            elif loss_key in ["asymmetric"]:
                kwargs["gamma_neg"] = params.get("gamma_neg", 4.0)
                kwargs["gamma_pos"] = params.get("gamma_pos", 1.0)
                kwargs["clip"] = params.get(
                    "asl_clip", 0.05
                )  # Conectado a modelos.toml

            loss_cls = LossFactory.get(loss_key) or nn.BCEWithLogitsLoss
            criterions[col] = loss_cls(**kwargs).to(device)

        return criterions

    @staticmethod
    def create_uncertainty_wrapper(
        tasks: list[str], device: torch.device
    ) -> MultiTaskUncertaintyLoss:
        """
        Instancia el contenedor de incertidumbre para el aprendizaje multi-tarea.
        """
        return MultiTaskUncertaintyLoss(tasks=tasks).to(device)
