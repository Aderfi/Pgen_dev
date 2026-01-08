# Pharmagen - Training Engine
# Unified Trainer Class.
# Handles Training, Validation, Metrics, and Checkpointing.


import logging
from collections.abc import Mapping, MutableSequence, Set
from typing import Any, cast

import torch
from optuna import Trial, TrialPruned
from torch import nn
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.config.manager import DIRS
from src.utils.checkpoint import CheckpointManager
from src.utils.exceptions import TrainingError
from src.utils.losses import MultiTaskUncertaintyLoss
from src.utils.memory import MemoryMonitor

logger = logging.getLogger(__name__)


class PGenTrainer:
    """
    Unified trainer for both standard training and Optuna optimization.

    Handles two execution contexts:
    1. Standard Training (from_optuna=False, trial=None):
       - Full checkpointing with CheckpointManager
       - Detailed logging and progress reporting
       - Memory monitoring per epoch
       - Model compilation for speed

    2. Optuna Trials (from_optuna=True, trial!=None):
       - Minimal logging (tuner handles progress)
       - No checkpointing (tuner manages best trials)
       - Lightweight memory cleanup
       - No model compilation (avoid overhead)

    Example:
        # Standard training
        >>> trainer = PGenTrainer(model, optimizer, .. ., from_optuna=False)
        >>> best_loss = trainer.fit(train_loader, val_loader, epochs=100, patience=10)

        # Optuna trial
        >>> trainer = PGenTrainer(model, optimizer, ..., from_optuna=True)
        >>> trial_loss = trainer.fit(... , trial=trial)
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        device: torch.device,
        target_cols: MutableSequence[str],
        multi_label_cols: Set[str],
        params:  Mapping[str, Any],
        uncertainty_module: MultiTaskUncertaintyLoss | None = None,
        from_optuna: bool = False,
    ):
        """Initialize PGenTrainer.

        Args:
            model: PyTorch model to train.
            optimizer:  Optimizer instance.
            scheduler: Learning rate scheduler.
            device: Device to train on (cuda/cpu).
            target_cols: List of target column names.
            multi_label_cols:  Set of multi-label column names.
            params:  Hyperparameter dictionary.
            uncertainty_module: Optional uncertainty loss module.
            from_optuna: If True, disables model compilation for faster trials.
        """
        self._validate_inputs(target_cols, multi_label_cols, device, params, from_optuna)

        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.target_cols = target_cols
        self.ml_cols = multi_label_cols
        self.params = params
        self.uncertainty_module = uncertainty_module
        self.from_optuna = from_optuna

        # Training utilities
        self.scaler = GradScaler()
        self.loss_fns = self._setup_criterions()
        self.best_loss = float("inf")
        self.patience_counter = 0
        self.current_epoch = 0

        # Model compilation (disabled during Optuna for speed)
        self.model = self._compile_model(model)
        if not from_optuna:
            self.checkpoint_manager = CheckpointManager(
                model_name="training_session",
                keep_last_n=3,
            )
            logger.debug("💾 CheckpointManager initialized (standard training)")
        else:
            self.checkpoint_manager = None
            logger.debug("🔬 Optuna mode - checkpointing delegated to tuner")

        DIRS["models"].mkdir(parents=True, exist_ok=True)

    def _validate_inputs(self, target_cols, multi_label_cols, device, params, from_optuna):
        """Validate constructor inputs."""
        if not target_cols:
            raise ValueError("target_cols cannot be empty")

        if not isinstance(multi_label_cols, set):
            raise TypeError(f"multi_label_cols must be set, got {type(multi_label_cols)}")

        if not isinstance(device, torch.device):
            raise TypeError(f"device must be torch.device, got {type(device)}")

        required_params = ["learning_rate", "weight_decay"]
        missing = [p for p in required_params if p not in params]
        if missing and not from_optuna:
            logger.warning(f"⚠️ Missing recommended params: {missing}")

    def _compile_model(self, model:  nn.Module) -> nn.Module:
        """Compile model with torch.compile (only in standard training).

        Rationale for skipping in Optuna:
        - Compilation overhead wasted on pruned trials
        - Multiple trials = multiple compilations
        - Optuna already parallelizes (n_jobs), compilation gives diminishing returns
        """
        if self.from_optuna:
            logger.debug("Skipping torch.compile in Optuna mode")
            return model

        try:
            compiled = torch.compile(
                model,
                mode="default",
                dynamic=True,
                backend="inductor",
            )
            logger.debug("⚡ Model compiled with torch. compile (inductor)")
            return cast(nn.Module, compiled)
        except Exception as e:
            logger.warning(f"⚠️ Compilation failed: {e}. Using eager mode.")
            return model

    def _setup_criterions(self) -> dict[str, nn.Module]:
        """Setup loss functions for each task."""
        from src.utils.module_builder import LossFactory

        return LossFactory.create_task_criterions(
            target_cols=self.target_cols,
            multi_label_cols=self.ml_cols,
            params=self.params,
            device=self.device,
        )

    def _compute_step(
        self, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Execute forward pass and compute loss for a batch.
        Expects batch from DoubleTowerCollater:
        {
            'drug_batch': PyG Batch object,
            'geno_batch': PyG Batch object,
            'targets': {'target_name':  Tensor, ...}
        }

        Args:
            batch: Batch dictionary from DataLoader.

        Returns:
            Tuple of (total_loss, metrics_dict).
        """
        # 1. Move graph data to device
        drug_data = batch["drug_batch"].to(self.device)
        geno_data = batch["geno_batch"].to(self.device)

        # 2. Move targets to device
        targets = {
            k: v.to(self.device)
            for k, v in batch["targets"].items()
            if k in self.target_cols
        }

        # 3. Forward pass
        outputs = self.model(drug_data, geno_data)

        # 4. Compute loss and metrics
        return self._calculate_loss_and_metrics(outputs, targets)

    def _calculate_loss_and_metrics(
        self, outputs: Mapping[str, torch.Tensor], targets: Mapping[str, torch. Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Calculate loss and accuracy metrics.

        Args:
            outputs: Model predictions {task:  logits}.
            targets: Ground truth labels {task: labels}.

        Returns:
            Tuple of (total_loss, metrics_dict).
        """
        # 1. Compute per-task losses
        losses_per_task = {}
        for t_col, t_true in targets.items():
            pred = outputs[t_col]
            # Type casting based on task type
            target = t_true.float() if t_col in self.ml_cols else t_true.long()
            losses_per_task[t_col] = self.loss_fns[t_col](pred, target)

        # 2. Aggregate losses
        if self.uncertainty_module:
            total_loss = self.uncertainty_module(losses_per_task)
        else:
            total_loss = sum(losses_per_task.values())

        total_loss = cast(torch.Tensor, total_loss)

        # 3. Compute diagnostic accuracy (not used for backprop)
        accuracies = []
        with torch.no_grad():
            for t_col, t_true in targets.items():
                pred = outputs[t_col]
                if t_col in self.ml_cols:
                    # Multi-label:  subset accuracy
                    probs = torch.sigmoid(pred)
                    preds_bin = (probs > 0.5).float() # noqa
                    acc = (preds_bin == t_true.float()).float().mean()
                else:
                    # Multi-class: top-1 accuracy
                    acc = (pred.argmax(1) == t_true.long()).float().mean()
                accuracies.append(acc.item())
        avg_acc = sum(accuracies) / len(accuracies) if accuracies else 0.0

        return total_loss, {"loss":  total_loss.item(), "acc": avg_acc}

    def train_epoch(self, loader: DataLoader) -> dict[str, float]:
        """
        Train for one epoch with mixed precision and gradient clipping.

        Args: Training DataLoader.     Returns: Dictionary of averaged metrics.
        """
        self.model.train()
        total_metrics = {"loss": 0.0, "acc": 0.0}
        n_batches = len(loader)
        if not self.from_optuna:
            MemoryMonitor.log_memory_stats(f"Epoch {self.current_epoch} start - ")

        # Progress bar (disabled during Optuna)
        progress_iteration = (
            loader if self.from_optuna else tqdm(loader, desc="Train", leave=False)
        )

        for batch_idx, batch in enumerate(progress_iteration):
            self.optimizer.zero_grad(set_to_none=True)

            # Mixed precision forward pass
            with autocast(device_type=self.device.type):
                loss, metrics = self._compute_step(batch)

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Accumulate metrics
            for k, v in metrics.items():
                total_metrics[k] += v

            # Periodic memory cleanup to prevent gradual accumulation
            if batch_idx % 50 == 0 and batch_idx > 0:
                MemoryMonitor.clear_memory(
                    device=self.device,
                    aggressive=not self.from_optuna
                )
        if not self.from_optuna:
            MemoryMonitor.log_memory_stats(f"Epoch {self.current_epoch} end - ")

        return {k:  v / n_batches for k, v in total_metrics.items()}

    def validate(self, loader: DataLoader) -> dict[str, float]:
        """Validate model on validation set."""
        self.model.eval()
        total_metrics = {"loss": 0.0, "acc": 0.0}
        n_batches = len(loader)

        with torch.inference_mode(), autocast(device_type=self.device.type):
            for batch in loader:
                _, metrics = self._compute_step(batch)
                for k, v in metrics.items():
                    total_metrics[k] += v

        return {k: v / n_batches for k, v in total_metrics.items()}

    def fit( #noqa
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        patience: int,
        trial: Trial | None = None,
    ) -> float:
        """Main training loop with context-aware behavior.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            epochs: Maximum number of epochs.
            patience: Early stopping patience.
            trial: Optional Optuna trial for HPO.

        Returns:
            Best validation loss achieved.

        Raises:
            TrainingError: If training fails in standard mode.
            Optuna.TrialPruned: If trial should be pruned.
        """
        if not trial and not self.from_optuna:
            # Standard training:  verbose logging
            logger.info(f"🏋️ Starting training on {self.device} for {epochs} epochs")
            logger.info(f"📊 Targets: {self.target_cols}")
            logger.info(f"⚙️ Patience: {patience}")
        elif trial:
            # Optuna trial: minimal logging
            logger.debug(f"Trial {trial.number}:  Training {epochs} epochs")

        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch

            # Training
            train_metrics = self.train_epoch(train_loader)
            train_loss = train_metrics["loss"]

            # ✅ NaN detection with context-aware error handling
            if torch.isnan(torch.tensor(train_loss)):
                msg = f"Training loss is NaN at epoch {epoch}"
                logger.error(f"❌ {msg}")

                if trial:
                    raise TrialPruned(msg)
                else:
                    raise TrainingError(msg)

            # Validation
            val_metrics = self.validate(val_loader)
            v_loss = val_metrics["loss"]

            # Learning rate scheduling
            is_plateau = isinstance(
                self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            )
            if is_plateau:
                self.scheduler.step(v_loss)
            else:
                self.scheduler.step()

            # ✅ Optuna reporting and pruning
            if trial:
                trial.report(v_loss, epoch)
                if trial.should_prune():
                    raise TrialPruned()

            # ✅ Early stopping with context-aware checkpointing
            if v_loss < self.best_loss:
                self.best_loss = v_loss
                self.patience_counter = 0

                # CRITICAL: Only checkpoint in standard training
                if self.checkpoint_manager:
                    self.checkpoint_manager.save_checkpoint(
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        epoch=epoch,
                        metrics={"val_loss": v_loss, "train_loss": train_loss},
                        uncertainty_module=self.uncertainty_module,
                        is_best=True,
                    )
                    logger.debug(f"💾 Checkpoint saved (epoch {epoch}, loss={v_loss:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    if not self.from_optuna:
                        logger.info(f"⏹️ Early stopping at epoch {epoch}/{epochs}")
                        logger.info(f"✅ Best val loss: {self.best_loss:.4f}")
                    else:
                        logger.debug(f"Trial early stop at epoch {epoch}")
                    break

        # ✅ Load best checkpoint (only in standard training)
        if self.checkpoint_manager:
            resume_info = self.checkpoint_manager.resume_training(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                uncertainty_module=self.uncertainty_module,
            )
            logger.info(
                f"✅ Loaded best checkpoint from epoch {resume_info['start_epoch']-1}"
            )

        return self.best_loss

    def _save_checkpoint(self, name: str):
        """Save model checkpoint."""
        filename = name if name.startswith("pmodel_") else f"pmodel_{name}"
        path = DIRS["models"] / filename

        state = {"model":  self.model.state_dict()}
        torch.save(state, path)
        logger.debug(f"Checkpoint saved:  {path}")


    def _load_checkpoint(self, name: str):
        """Load model checkpoint."""
        filename = name if name.startswith("pmodel_") else f"pmodel_{name}"
        path = DIRS["models"] / filename

        if path.exists():
            state = torch.load(path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state["model"])
            logger.debug(f"Checkpoint loaded: {path}")
        else:
            logger.warning(f"Checkpoint not found: {path}")
