"""Standard (non-Optuna) trainer.

Uses ``CheckpointManager`` for best-checkpoint persistence, runs
``torch.compile`` on construction, and renders a tqdm progress bar per
epoch. Use directly:

    from src.model.training import StandardTrainer

    trainer = StandardTrainer(model, optimizer, scheduler, ...)
    best_loss = trainer.fit(train_loader, val_loader, epochs=100, patience=10)
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, MutableSequence, Set
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.config import get_settings
from src.model.checkpoint import CheckpointManager
from src.model.training.loop import TrainingLoop

if TYPE_CHECKING:
    from src.model.losses import CompositionalLabelLoss, MultiTaskLoss

logger = logging.getLogger(__name__)


class StandardTrainer(TrainingLoop):
    """The full-featured trainer used by the production pipeline."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        device: torch.device,
        target_cols: MutableSequence[str],
        multi_label_cols: Set[str],
        params: Mapping[str, Any],
        multitask_loss: MultiTaskLoss,
        compose_loss: CompositionalLabelLoss | None = None,
        compose_weight: float = 0.5,
        *,
        checkpoint_name: str = "training_session",
        keep_last_n: int = 3,
    ) -> None:
        super().__init__(
            model,
            optimizer,
            scheduler,
            device,
            target_cols,
            multi_label_cols,
            params,
            multitask_loss,
            compose_loss,
            compose_weight,
        )
        get_settings().paths.models.mkdir(parents=True, exist_ok=True)
        self.checkpoint_manager = CheckpointManager(
            model_name=checkpoint_name,
            keep_last_n=keep_last_n,
        )
        logger.debug(
            "CheckpointManager ready (name=%s, keep_last=%d).",
            checkpoint_name,
            keep_last_n,
        )

    # ----------------------------------------------------------- hooks

    def _maybe_compile(self, model: nn.Module) -> nn.Module:
        """Compile the model with ``torch.compile`` for inference speed."""
        try:
            compiled = torch.compile(
                model, mode="default", dynamic=True, backend="inductor"
            )
            logger.debug("Model compiled with torch.compile (inductor).")
            return cast("nn.Module", compiled)
        except Exception as e:  # noqa: BLE001
            logger.warning("torch.compile failed: %s. Using eager mode.", e)
            return model

    def _iter_train_batches(self, loader: DataLoader) -> Iterable[Any]:
        return tqdm(loader, desc="Train", leave=False)

    # ------------------------------------------------------------ fit

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        patience: int,
        **kwargs: Any,  # noqa: ARG002 — kept for signature compatibility with TrainingLoop.fit
    ) -> float:
        logger.info("Starting training on %s for %d epochs.", self.device, epochs)
        logger.info("Targets: %s. Patience: %d.", self.target_cols, patience)

        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch

            train_metrics = self.train_epoch(train_loader)
            train_loss = train_metrics["loss"]
            self._check_nan(train_loss, epoch)

            val_metrics = self.validate(val_loader)
            v_loss = val_metrics["loss"]

            self._step_scheduler(v_loss)

            if v_loss < self.best_loss:
                self.best_loss = v_loss
                self.patience_counter = 0
                self.checkpoint_manager.save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    metrics={"val_loss": v_loss, "train_loss": train_loss},
                    uncertainty_module=self.multitask_loss,
                    is_best=True,
                )
                logger.debug(
                    "Checkpoint saved at epoch %d (val_loss=%.4f).", epoch, v_loss
                )
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    logger.info(
                        "Early stopping at epoch %d/%d. Best val loss: %.4f",
                        epoch,
                        epochs,
                        self.best_loss,
                    )
                    break

        # Reload best checkpoint so the model returned to the caller is the best one.
        resume_info = self.checkpoint_manager.resume_training(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            uncertainty_module=self.multitask_loss,
        )
        logger.info(
            "Loaded best checkpoint from epoch %d.", resume_info["start_epoch"] - 1
        )
        return self.best_loss
