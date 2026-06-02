"""Shared training-loop primitives.

``TrainingLoop`` owns the parts that don't change between standard and
Optuna runs: input validation, criterion setup, the per-step forward
+ backward, validation, and epoch metric aggregation.

Subclasses customize:

* ``_should_compile`` — whether ``torch.compile`` runs at __init__.
* ``_iter_train_batches`` — wraps the loader (e.g. with tqdm).
* ``_on_epoch_end`` — receives metrics for the just-finished epoch and
  returns a stop signal (True = stop early).
* ``_on_finish`` — called after the training loop exits cleanly.
* ``fit`` — kept abstract so subclasses can shape the public API
  (e.g. accept an Optuna ``trial`` argument).
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, MutableSequence, Set
from typing import Any, cast

import torch
from torch import nn
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader

from src.core import TrainingError
from src.model.losses import MultiTaskUncertaintyLoss

logger = logging.getLogger(__name__)


class TrainingLoop(ABC):
    """Abstract base class for trainers.

    Subclasses must implement :meth:`fit`. Everything else is shared and
    parametrized via small protected hooks.
    """

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
        self._validate_inputs(target_cols, multi_label_cols, device, params)

        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.target_cols = list(target_cols)
        self.ml_cols = set(multi_label_cols)
        self.params = dict(params)
        self.uncertainty_module = uncertainty_module

        self.scaler = GradScaler()
        self.loss_fns = self._setup_criterions()

        self.best_loss = float("inf")
        self.patience_counter = 0
        self.current_epoch = 0

        # Subclasses opt-in to torch.compile via the hook.
        self.model = self._maybe_compile(model)

    # ------------------------------------------------------------------ hooks

    def _maybe_compile(self, model: nn.Module) -> nn.Module:
        """Override to enable / disable ``torch.compile`` for this trainer."""
        return model

    def _iter_train_batches(self, loader: DataLoader) -> Iterable[Any]:
        """Override to wrap the iterator (e.g. with tqdm)."""
        return loader

    def _on_epoch_end(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
    ) -> bool:
        """Called after each epoch. Return True to stop training early."""
        return False

    def _on_finish(self) -> None:
        """Called once the training loop exits cleanly (not via exception)."""
        return

    # ----------------------------------------------------------------- shared

    @staticmethod
    def _validate_inputs(
        target_cols: Iterable[str],
        multi_label_cols: object,
        device: object,
        params: object,
    ) -> None:
        if not list(target_cols):
            msg = "target_cols cannot be empty"
            raise ValueError(msg)
        if not isinstance(multi_label_cols, (set, frozenset)):
            msg = (
                f"multi_label_cols must be a set, got {type(multi_label_cols).__name__}"
            )
            raise TypeError(msg)
        if not isinstance(device, torch.device):
            msg = f"device must be torch.device, got {type(device).__name__}"
            raise TypeError(msg)
        if not isinstance(params, Mapping):
            msg = f"params must be a Mapping, got {type(params).__name__}"
            raise TypeError(msg)

    def _setup_criterions(self) -> dict[str, nn.Module]:
        from src.model.factories import LossFactory

        return LossFactory.create_task_criterions(
            target_cols=self.target_cols,
            multi_label_cols=self.ml_cols,
            params=self.params,
            device=self.device,
        )

    # ------------------------------------------------------- forward / metric

    def _compute_step(
        self, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        drug_data = batch["drug_batch"].to(self.device)
        geno_data = batch["geno_batch"].to(self.device)
        targets = {
            k: v.to(self.device)
            for k, v in batch["targets"].items()
            if k in self.target_cols
        }
        outputs = self.model(drug_data, geno_data)
        return self._calculate_loss_and_metrics(outputs, targets)

    def _calculate_loss_and_metrics(
        self,
        outputs: Mapping[str, torch.Tensor],
        targets: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        losses_per_task = {}
        for t_col, t_true in targets.items():
            pred = outputs[t_col]
            target = t_true.float() if t_col in self.ml_cols else t_true.long()
            losses_per_task[t_col] = self.loss_fns[t_col](pred, target)

        if self.uncertainty_module:
            total_loss = self.uncertainty_module(losses_per_task)
        else:
            total_loss = sum(losses_per_task.values())
        total_loss = cast("torch.Tensor", total_loss)

        accuracies: list[float] = []
        with torch.no_grad():
            for t_col, t_true in targets.items():
                pred = outputs[t_col]
                if t_col in self.ml_cols:
                    probs = torch.sigmoid(pred)
                    preds_bin = (probs > 0.5).float()
                    acc = (preds_bin == t_true.float()).float().mean()
                else:
                    acc = (pred.argmax(1) == t_true.long()).float().mean()
                accuracies.append(acc.item())
        avg_acc = sum(accuracies) / len(accuracies) if accuracies else 0.0
        return total_loss, {"loss": total_loss.item(), "acc": avg_acc}

    # ----------------------------------------------------- public step + val

    def train_epoch(self, loader: DataLoader) -> dict[str, float]:
        """One epoch of training with mixed-precision + gradient clipping."""
        self.model.train()
        total = {"loss": 0.0, "acc": 0.0}
        n_batches = len(loader)
        for batch in self._iter_train_batches(loader):
            self.optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=self.device.type):
                loss, metrics = self._compute_step(batch)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            for k, v in metrics.items():
                total[k] += v
        return {k: v / n_batches for k, v in total.items()}

    def validate(self, loader: DataLoader) -> dict[str, float]:
        """One epoch of validation (no grad, no scaler)."""
        self.model.eval()
        total = {"loss": 0.0, "acc": 0.0}
        n_batches = len(loader)
        with torch.inference_mode(), autocast(device_type=self.device.type):
            for batch in loader:
                _, metrics = self._compute_step(batch)
                for k, v in metrics.items():
                    total[k] += v
        return {k: v / n_batches for k, v in total.items()}

    # ----------------------------------------------------- step the scheduler

    def _step_scheduler(self, val_loss: float) -> None:
        is_plateau = isinstance(
            self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
        )
        if is_plateau:
            self.scheduler.step(val_loss)
        else:
            self.scheduler.step()

    # ----------------------------------------------------- public API

    @staticmethod
    def _check_nan(value: float, epoch: int) -> None:
        """Raise TrainingError if a loss is NaN — subclasses may translate
        this to ``optuna.TrialPruned`` instead.
        """
        if math.isnan(value):
            msg = f"Training loss is NaN at epoch {epoch}"
            logger.error(msg)
            raise TrainingError(msg)

    @abstractmethod
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        patience: int,
        **kwargs: Any,
    ) -> float:
        """Run the training loop and return the best validation loss seen."""
