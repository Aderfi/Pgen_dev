"""Training loop for the two-tower GNN.

One :class:`Trainer` serves both standard training and Optuna trials; the
differences are plain constructor/``fit`` options instead of subclasses:

* ``checkpoint`` — a :class:`CheckpointManager` persists every improvement and
  the best weights are reloaded at the end. ``None`` (Optuna trials) keeps
  everything in memory; the tuner tracks the best trial itself.
* ``compile_model`` — wrap the model with ``torch.compile`` (worth it for long
  runs, wasted on pruned trials).
* ``show_progress`` — per-epoch tqdm bar.
* ``fit(..., trial=...)`` — report validation losses to an Optuna trial for
  pruning; a NaN loss then prunes the trial instead of raising.

    trainer = Trainer(model, optimizer, scheduler, device, ["phenotype_category"],
                      set(), multitask_loss, checkpoint=CheckpointManager("TwoTowerGAT"))
    best_val_loss = trainer.fit(train_loader, val_loader, epochs=100, patience=10)
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import nn
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

from src.core import TrainingError

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Set

    from optuna import Trial
    from torch.utils.data import DataLoader

    from src.model.checkpoint import CheckpointManager
    from src.model.losses import CompositionalLabelLoss, MultiTaskLoss

logger = logging.getLogger(__name__)

GRAD_CLIP_NORM = 1.0


class Trainer:
    """Mixed-precision training loop with early stopping.

    ``model`` is called as ``model(drug_batch, geno_batch)`` and must return a
    dict of per-axis logits (plus ``"_z"`` when it has a compose head).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        device: torch.device,
        target_cols: Iterable[str],
        multi_label_cols: Set[str],
        multitask_loss: MultiTaskLoss,
        compose_loss: CompositionalLabelLoss | None = None,
        compose_weight: float = 0.5,
        *,
        checkpoint: CheckpointManager | None = None,
        compile_model: bool = False,
        show_progress: bool = False,
    ) -> None:
        self.target_cols = list(target_cols)
        if not self.target_cols:
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

        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.ml_cols = set(multi_label_cols)
        self.multitask_loss = multitask_loss
        self.compose_loss = compose_loss
        self.compose_weight = compose_weight
        self.checkpoint = checkpoint
        self.show_progress = show_progress

        self.scaler = GradScaler()
        self.best_loss = float("inf")
        self.patience_counter = 0
        self.current_epoch = 0

        # Uncompiled reference for structural access (`.compose`,
        # `.axis_heads`) — torch.compile wraps the model in an OptimizedModule.
        self._raw_model = model
        self.model = _compile(model) if compile_model else model

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
        total_loss, _per_task = self.multitask_loss(dict(outputs), dict(targets))
        total_loss = cast("torch.Tensor", total_loss)

        compose = getattr(self._raw_model, "compose", None)
        if self.compose_loss is not None and compose is not None and "_z" in outputs:
            composable = self._raw_model.axis_heads.single_label_axes()
            if composable:
                tuples = torch.stack(
                    [targets[a].long().view(-1) for a in composable], dim=1
                )
                target_emb = compose.embed_tuples(
                    tuples, self._raw_model.axis_heads.axis_embeddings
                )
                total_loss = total_loss + self.compose_weight * self.compose_loss(
                    outputs["_z"], target_emb
                )

        accuracies: list[float] = []
        with torch.no_grad():
            for col, truth in targets.items():
                pred = outputs[col]
                if col in self.ml_cols:
                    hits = (torch.sigmoid(pred) > 0.5).float() == truth.float()
                else:
                    hits = pred.argmax(1) == truth.long()
                accuracies.append(hits.float().mean().item())
        avg_acc = sum(accuracies) / len(accuracies) if accuracies else 0.0
        return total_loss, {"loss": total_loss.item(), "acc": avg_acc}

    # ----------------------------------------------------- epochs

    def train_epoch(self, loader: DataLoader) -> dict[str, float]:
        """One epoch of training with mixed precision + gradient clipping."""
        self.model.train()
        total = {"loss": 0.0, "acc": 0.0}
        batches: Iterable[Any] = loader
        if self.show_progress:
            from tqdm.auto import tqdm

            batches = tqdm(loader, desc="Train", leave=False)
        for batch in batches:
            self.optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=self.device.type):
                loss, metrics = self._compute_step(batch)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=GRAD_CLIP_NORM)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            for k, v in metrics.items():
                total[k] += v
        return {k: v / len(loader) for k, v in total.items()}

    def validate(self, loader: DataLoader) -> dict[str, float]:
        """One validation pass (no grad, no scaler)."""
        self.model.eval()
        total = {"loss": 0.0, "acc": 0.0}
        with torch.inference_mode(), autocast(device_type=self.device.type):
            for batch in loader:
                _, metrics = self._compute_step(batch)
                for k, v in metrics.items():
                    total[k] += v
        return {k: v / len(loader) for k, v in total.items()}

    def _step_scheduler(self, val_loss: float) -> None:
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(val_loss)
        else:
            self.scheduler.step()

    # ----------------------------------------------------- public API

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        patience: int,
        *,
        trial: Trial | None = None,
    ) -> float:
        """Train until ``epochs`` or early stop; return the best validation loss."""
        logger.info(
            "Training on %s for %d epochs (patience=%d, targets=%s)",
            self.device,
            epochs,
            patience,
            self.target_cols,
        )
        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            train_loss = self.train_epoch(train_loader)["loss"]
            self._check_nan(train_loss, epoch, pruning=trial is not None)

            val_metrics = self.validate(val_loader)
            val_loss = val_metrics["loss"]
            self._step_scheduler(val_loss)
            logger.info(
                "Epoch %d/%d: train_loss=%.4f val_loss=%.4f val_acc=%.4f",
                epoch,
                epochs,
                train_loss,
                val_loss,
                val_metrics["acc"],
            )

            if trial is not None:
                trial.report(val_loss, epoch)
                if trial.should_prune():
                    from optuna import TrialPruned

                    raise TrialPruned

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.patience_counter = 0
                if self.checkpoint is not None:
                    self.checkpoint.save_checkpoint(
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        epoch=epoch,
                        metrics={"val_loss": val_loss, "train_loss": train_loss},
                        uncertainty_module=self.multitask_loss,
                        is_best=True,
                    )
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    logger.info(
                        "Early stopping at epoch %d/%d (best val_loss=%.4f)",
                        epoch,
                        epochs,
                        self.best_loss,
                    )
                    break

        if self.checkpoint is not None and self.best_loss < float("inf"):
            # Hand the caller the best weights, not the last ones.
            info = self.checkpoint.resume_training(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                uncertainty_module=self.multitask_loss,
            )
            logger.info("Loaded best checkpoint from epoch %d", info["start_epoch"] - 1)
        return self.best_loss

    @staticmethod
    def _check_nan(value: float, epoch: int, *, pruning: bool = False) -> None:
        """Raise on a NaN loss: ``TrialPruned`` under Optuna, else ``TrainingError``."""
        if not math.isnan(value):
            return
        msg = f"Training loss is NaN at epoch {epoch}"
        logger.error(msg)
        if pruning:
            from optuna import TrialPruned

            raise TrialPruned(msg)
        raise TrainingError(msg)


def _compile(model: nn.Module) -> nn.Module:
    try:
        compiled = torch.compile(
            model, mode="default", dynamic=True, backend="inductor"
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("torch.compile failed: %s. Using eager mode.", e)
        return model
    logger.debug("Model compiled with torch.compile (inductor)")
    return cast("nn.Module", compiled)


__all__ = ["Trainer"]
