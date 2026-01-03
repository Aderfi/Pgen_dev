# Pharmagen - Training Engine
# Unified Trainer Class.
# Handles Training, Validation, Metrics, and Checkpointing.

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import optuna
import torch
import torch.nn as nn
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.config.manager import DIRS
from src.utils.losses import MultiTaskUncertaintyLoss

logger = logging.getLogger(__name__)

class PGenTrainer:
    """
    Handles the training lifecycle of the PharmagenTwoTower model.
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        device: torch.device,
        target_cols: List[str],
        multi_label_cols: Set[str],
        params: Dict[str, Any],
        uncertainty_module: Optional[MultiTaskUncertaintyLoss] = None,
        from_optuna: bool = False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.target_cols = target_cols
        self.ml_cols = multi_label_cols
        self.params = params
        self.uncertainty_module = uncertainty_module
        
        self.scaler = GradScaler()
        self.loss_fns = self._setup_criterions()
        self.best_loss = float("inf")
        self.patience_counter = 0
        self.from_optuna = from_optuna

        # Ensure model directory exists
        DIRS["models"].mkdir(parents=True, exist_ok=True)

        from src.utils.module_builder import LossFactory
        self.loss_fns = LossFactory.create_task_criterions(
            target_cols=target_cols,
            multi_label_cols=multi_label_cols,
            params=params,
            device=device
        )

    def _setup_criterions(self) -> Dict[str, nn.Module]:
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

    def _compute_step(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
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

    def _calculate_loss_and_metrics(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Tuple[Any, Dict[str, float]]:
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
                    preds_bin = (probs > 0.5).float()
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
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            for k, v in metrics.items():
                total_metrics[k] += v
            
        return {k: v / n_batches for k, v in total_metrics.items()}

    def validate(self, loader: DataLoader) -> Dict[str, float]:
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

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int, patience: int, trial: optuna.Trial | None) -> float:
        # Standard fit loop (Same as provided, just ensuring it calls the updated train_epoch)
        if not self.from_optuna:
            logger.info(f" Starting training on {self.device} for {epochs} epochs.")
        
        for epoch in range(1, epochs + 1):
            t_metrics = self.train_epoch(train_loader)
            v_metrics = self.validate(val_loader)
            
            v_loss = v_metrics["loss"]
            
            if not self.from_optuna:
                logger.info(
                    f" Epoch {epoch:02d} | "
                    f"Train Loss: {t_metrics['loss']:.4f} | "
                    f"Val Loss: {v_loss:.4f}"
                )
            
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(v_loss)
            else:
                self.scheduler.step()

            if trial:
                trial.report(v_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            if v_loss < self.best_loss:
                self.best_loss = v_loss
                self.patience_counter = 0
                self._save_checkpoint("best_model.pth")
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    logger.info("Early stopping triggered.")
                    break
        
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