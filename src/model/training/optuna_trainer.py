"""Optuna-trial trainer.

Skips ``torch.compile`` (overhead is wasted on pruned trials), skips
checkpointing (the tuner manages best-trial selection), and reports
intermediate validation losses to the Optuna trial for pruning.

NaN losses become ``optuna.TrialPruned`` rather than ``TrainingError`` so
the study continues sampling other trials.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, MutableSequence, Set
from typing import Any

import torch
from optuna import Trial, TrialPruned
from torch import nn
from torch.utils.data import DataLoader

from src.model.training.loop import TrainingLoop
from src.model.losses import MultiTaskUncertaintyLoss

logger = logging.getLogger(__name__)


class OptunaTrialTrainer(TrainingLoop):
    """Trainer used by the Optuna tuner — minimal logging, no checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        device: torch.device,
        target_cols: MutableSequence[str],
        multi_label_cols: Set[str],
        params: Mapping[str, Any],
        uncertainty_module: MultiTaskUncertaintyLoss | None = None,
    ) -> None:
        super().__init__(
            model, optimizer, scheduler, device, target_cols,
            multi_label_cols, params, uncertainty_module,
        )
        # No checkpoint manager — the tuner persists best trials separately.

    # No torch.compile, no tqdm — both inherit the no-op base behaviour.

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        patience: int,
        *,
        trial: Trial | None = None,
        **kwargs: Any,  # noqa: ARG002 — keep base signature
    ) -> float:
        if trial is not None:
            logger.debug("Trial %d: training %d epochs.", trial.number, epochs)

        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch

            train_metrics = self.train_epoch(train_loader)
            train_loss = train_metrics["loss"]

            # NaN → prune (don't crash the whole study).
            if math.isnan(train_loss):
                msg = f"Training loss is NaN at epoch {epoch}"
                logger.warning(msg)
                raise TrialPruned(msg)

            val_metrics = self.validate(val_loader)
            v_loss = val_metrics["loss"]

            self._step_scheduler(v_loss)

            if trial is not None:
                trial.report(v_loss, epoch)
                if trial.should_prune():
                    raise TrialPruned()

            if v_loss < self.best_loss:
                self.best_loss = v_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    logger.debug("Trial early stop at epoch %d.", epoch)
                    break

        return self.best_loss
