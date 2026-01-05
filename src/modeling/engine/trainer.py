# Pharmagen - Training Engine
# Unified Trainer Class.
# Handles Training, Validation, Metrics, and Checkpointing.

import logging
import math
from dataclasses import dataclass
from typing import Any, cast

import optuna
import torch
from torch import nn
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.config.manager import DIRS
from src.utils.losses import MultiTaskUncertaintyLoss

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Configuration for PGenTrainer to reduce parameter count."""

    device: torch.device
    target_cols: list[str]
    multi_label_cols: set[str]
    params: dict[str, Any]
    from_optuna: bool = False

class PGenTrainer:
    """
    Handles the training lifecycle of the PharmagenTwoTower model.
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        config: TrainerConfig,
        uncertainty_module: MultiTaskUncertaintyLoss | None = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = config.device
        self.target_cols = config.target_cols
        self.ml_cols = config.multi_label_cols
        self.params = config.params
        self.uncertainty_module = uncertainty_module

        self.scaler = GradScaler()
        self.best_loss = float("inf")
        self.patience_counter = 0
        self.from_optuna = config.from_optuna

        if not config.from_optuna:
            compiled_model = torch.compile(
                model,
                mode="default",
                dynamic=True,
                backend="inductor" # Explícito para Linux/Debian
                )
            self.model = cast(nn.Module, compiled_model)

        # Ensure model directory exists
        DIRS["models"].mkdir(parents=True, exist_ok=True)

        # Setup loss functions using factory
        self.loss_fns = self._setup_criterions()

    def _setup_criterions(self) -> dict[str, nn.Module]:
        """
        Refactorización: Delegación a Factoría.
        Ahora utiliza la lógica híbrida (Asymmetric/Focal) y los parámetros de Optuna.
        """
        from src.utils.module_builder import LossFactory

        # 1. Utilizamos la factoría para obtener el diccionario de pérdidas configurado
        # Esto inyecta automáticamente 'gamma' y 'asl_clip' desde self.params
        return LossFactory.create_task_criterions(
            target_cols=self.target_cols,
            multi_label_cols=self.ml_cols,
            params=self.params,
            device=self.device
        )

    def _compute_step(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Computes forward pass, loss, and metrics for the Dual Graph model.
        Expects batch dict from DoubleTowerCollater:
        {
            'drug_batch': Batch object,
            'haplo_batch': Batch object,
            'targets': { 'target_name': Tensor, ... }
        }
        """
        # 1. Unpack and Move Graphs to Device
        # The collator provides 'drug_batch' and 'haplo_batch' which are PyG Batch objects
        drug_data = batch["drug_batch"].to(self.device)
        haplo_data = batch["haplo_batch"].to(self.device)

        # 2. Unpack and Move Targets
        # Targets are already stacked by the collator, just need to move to device
        targets = {
            k: v.to(self.device)
            for k, v in batch["targets"].items()
            if k in self.target_cols
        }

        # 3. Forward Pass
        # Matches PharmagenTwoTower.forward(drug_data, haplo_data)
        outputs = self.model(drug_data, haplo_data)

        # 4. Loss & Metrics
        return self._calculate_loss_and_metrics(outputs, targets) # type: ignore

    def _calculate_loss_and_metrics(self, outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]) -> tuple[Any, dict[str, float]]:
        """Shared loss and metric calculation."""
        # 1. Compute Losses
        losses_per_task = {}
        for t_col, t_true in targets.items():
            pred = outputs[t_col]

            target = t_true.float() if t_col in self.ml_cols else t_true.long()

            losses_per_task[t_col] = self.loss_fns[t_col](pred, target)

        # Aggregate Loss
        if self.uncertainty_module:
            total_loss = self.uncertainty_module(losses_per_task)
        else:
            # Simple sum (can add weighted sum logic here if needed)
            total_loss = sum(losses_per_task.values())

        # 2. Compute Basic Accuracy (Diagnostic only)
        accuracies = []
        with torch.no_grad():
            for t_col, t_true in targets.items():
                pred = outputs[t_col]
                if t_col in self.ml_cols:
                    # Multi-label accuracy (subset match or simple binary match)
                    probs = torch.sigmoid(pred)
                    preds_bin = (probs > (1/2)).float()
                    acc = (preds_bin == t_true.float()).float().mean()
                else:
                    # Multi-class accuracy
                    acc = (pred.argmax(1) == t_true.long()).float().mean()
                accuracies.append(acc.item())

        avg_acc = sum(accuracies) / len(accuracies) if accuracies else 0.0

        return total_loss, {"loss": total_loss.item(), "acc": avg_acc} # type: ignore

    def train_epoch(self, loader: DataLoader) -> dict[str, float]:
        self.model.train()
        total_metrics = {"loss": 0.0, "acc": 0.0}
        n_batches = len(loader)

        progress_iteration = loader if self.from_optuna else tqdm(loader, desc="Train", leave=False)

        for batch in progress_iteration:
            self.optimizer.zero_grad(set_to_none=True)

            # Use autocast for GATv2 mixed precision (Faster & Less Memory)
            with autocast(device_type=self.device.type):
                loss, metrics = self._compute_step(batch)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            for k, v in metrics.items():
                total_metrics[k] += v

        return {k: v / n_batches for k, v in total_metrics.items()}

    def validate(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        total_metrics = {"loss": 0.0, "acc": 0.0}
        n_batches = len(loader)

        # torch.inference_mode() es más rápido que no_grad()
        with torch.inference_mode(), autocast(device_type=self.device.type):
            for batch in loader:
                _, metrics = self._compute_step(batch)
                for k, v in metrics.items():
                    total_metrics[k] += v

        return {k: v / n_batches for k, v in total_metrics.items()}

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int, patience: int, trial: optuna.Trial | None = None) -> float:
        if not trial:
            logger.info(f"Starting training on {self.device} for {epochs} epochs.")

        for epoch in range(1, epochs + 1):
            # 1. Entrenamiento y extracción segura de valor (pattern matching de tipos implícito)
            raw_loss = self.train_epoch(train_loader)["loss"]
            train_loss = raw_loss.item() if isinstance(raw_loss, torch.Tensor) else raw_loss

            # 2. Manejo de NaN condensado (Guard Clause)
            if math.isnan(train_loss):
                logger.warning(f"NaN loss at epoch {epoch}. {'Pruning' if trial else 'Returning inf'}")
                if trial:
                    raise optuna.TrialPruned("Loss is NaN")
                return float("inf")

            # 3. Validación
            v_loss = self.validate(val_loader)["loss"]

            # 4. Scheduler (Expresión condicional en una línea)
            is_plateau = isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
            self.scheduler.step(v_loss) if is_plateau else self.scheduler.step()

            # 5. Optuna Reporting
            if trial:
                trial.report(v_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            # 6. Early Stopping y Checkpointing (Lógica simplificada)
            if v_loss < self.best_loss:
                self.best_loss, self.patience_counter = v_loss, 0 # Asignación múltiple
                if not self.from_optuna:
                    self._save_checkpoint("best_model.pth")
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    logger.info("Early stopping triggered.")
                    break

        if not self.from_optuna:
            self._load_checkpoint("best_model.pth")
        return self.best_loss

    def _save_checkpoint(self, name: str):
        path = DIRS["models"] / f"pmodel_{name}" if not name.startswith("pmodel_") else DIRS["models"] / name
        state = {"model": self.model.state_dict()}
        torch.save(state, path)

    def _load_checkpoint(self, name: str):
        path = DIRS["models"] / f"pmodel_{name}" if not name.startswith("pmodel_") else DIRS["models"] / name
        if path.exists():
            state = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state["model"])
