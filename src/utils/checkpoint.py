# Pharmagen - Checkpoint Management
# Handles saving and loading training state for resuming.

"""
Checkpoint Manager for Training Resumption.

This module provides utilities for saving and loading training checkpoints,
enabling interrupted training sessions to be resumed seamlessly.

Features:
- Atomic checkpoint saving (prevents corruption)
- Best model tracking
- Automatic cleanup of old checkpoints
- Comprehensive state persistence
- Validation of checkpoint integrity
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from src.config. manager import DIRS
from src.utils.exceptions import ModelError

logger = logging.getLogger(__name__)

# Constants
CHECKPOINT_DIR = DIRS["models"] / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_NAME = "best_model.pt"
LATEST_CHECKPOINT_NAME = "latest_checkpoint.pt"
CHECKPOINT_INFO_NAME = "checkpoint_info.json"


class CheckpointManager:
    """
    Manages training checkpoints with automatic saving and loading.

    Handles:
    - Model state dict
    - Optimizer state
    - Scheduler state
    - Training metrics
    - Epoch tracking
    - Random states for reproducibility

    Example:
        >>> manager = CheckpointManager("TwoTowerGAT")
        >>> manager.save_checkpoint(model, optimizer, epoch=10, metrics={"loss": 0.5})
        >>> state = manager.load_checkpoint()
        >>> model. load_state_dict(state["model_state_dict"])
    """

    def __init__(
        self,
        model_name: str,
        save_dir: Optional[Path] = None,
        keep_last_n: int = 3,
    ):
        """
        Initialize checkpoint manager.

        Args:
            model_name: Name of the model (used for checkpoint directory).
            save_dir: Custom save directory (defaults to CHECKPOINT_DIR/model_name).
            keep_last_n: Number of recent checkpoints to keep (0 = keep all).
        """
        self.model_name = model_name
        self.save_dir = save_dir or (CHECKPOINT_DIR / model_name)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n

        logger.info(f"CheckpointManager initialized for '{model_name}'")
        logger.info(f"Checkpoint directory: {self.save_dir}")

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        scheduler: Optional[torch.optim. lr_scheduler._LRScheduler] = None,
        uncertainty_module: Optional[torch.nn.Module] = None,
        is_best:  bool = False,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save a complete training checkpoint.

        Args:
            model: PyTorch model to save.
            optimizer:  Optimizer state.
            epoch: Current epoch number.
            metrics: Dictionary of metrics (e.g., {"val_loss": 0.5, "val_acc": 0.9}).
            scheduler: Optional learning rate scheduler.
            uncertainty_module: Optional uncertainty estimation module.
            is_best: Whether this is the best model so far.
            extra_state: Additional custom state to save.

        Returns:
            Path to saved checkpoint file.
        """
        checkpoint = {
            # Model and training state
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,

            # Optional components
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "uncertainty_state_dict": (
                uncertainty_module.state_dict() if uncertainty_module else None
            ),

            # Metadata
            "model_name": self.model_name,
            "timestamp": datetime. now().isoformat(),

            # Reproducibility
            "random_state":  {
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": (
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                ),
            },

            # Extra custom state
            "extra_state":  extra_state or {},
        }

        # Save latest checkpoint
        latest_path = self.save_dir / LATEST_CHECKPOINT_NAME
        self._atomic_save(checkpoint, latest_path)
        logger.info(f"💾 Saved checkpoint at epoch {epoch} to {latest_path}")

        # Save epoch-specific checkpoint
        epoch_path = self.save_dir / f"checkpoint_epoch_{epoch: 04d}.pt"
        self._atomic_save(checkpoint, epoch_path)
        logger.debug(f"Saved epoch checkpoint:  {epoch_path}")

        # Save best model if applicable
        if is_best:
            best_path = self.save_dir / BEST_MODEL_NAME
            self._atomic_save(checkpoint, best_path)
            logger.info(f"⭐ New best model saved with metrics: {metrics}")

        # Save checkpoint info (JSON for easy inspection)
        self._save_checkpoint_info(epoch, metrics, is_best)

        # Cleanup old checkpoints
        if self.keep_last_n > 0:
            self._cleanup_old_checkpoints()

        return latest_path

    def load_checkpoint(
        self,
        checkpoint_path: Optional[Path] = None,
        load_best: bool = False,
    ) -> Dict[str, Any]:
        """
        Load a checkpoint from disk.

        Args:
            checkpoint_path: Path to specific checkpoint file.
                If None, loads the latest checkpoint.
            load_best: If True, loads the best model instead of latest.

        Returns:
            Dictionary containing all saved state.

        Raises:
            ModelError: If checkpoint not found or corrupted.
        """
        # Determine which checkpoint to load
        if checkpoint_path:
            path = Path(checkpoint_path)
        elif load_best:
            path = self.save_dir / BEST_MODEL_NAME
        else:
            path = self.save_dir / LATEST_CHECKPOINT_NAME

        if not path.exists():
            raise ModelError(f"Checkpoint not found:  {path}")

        try:
            checkpoint = torch.load(path, map_location="cpu")
            logger.info(f"✅ Loaded checkpoint from {path}")
            logger.info(f"Checkpoint epoch: {checkpoint. get('epoch', 'unknown')}")
            logger.info(f"Checkpoint metrics: {checkpoint.get('metrics', {})}")

            return checkpoint

        except Exception as e:
            raise ModelError(f"Failed to load checkpoint from {path}: {e}") from e

    def resume_training(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        uncertainty_module: Optional[torch.nn.Module] = None,
        checkpoint_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Resume training from a checkpoint.

        Loads all saved state into the provided objects and returns metadata.

        Args:
            model: Model to load state into.
            optimizer: Optimizer to load state into.
            scheduler: Optional scheduler to load state into.
            uncertainty_module: Optional uncertainty module to load state into.
            checkpoint_path: Path to checkpoint (defaults to latest).

        Returns:
            Dictionary with resume information (epoch, metrics, etc.).

        Raises:
            ModelError: If checkpoint loading fails.
        """
        checkpoint = self.load_checkpoint(checkpoint_path)

        # Load model state
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("✅ Restored model state")

        # Load optimizer state
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info("✅ Restored optimizer state")

        # Load scheduler state if available
        if scheduler and checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            logger.info("✅ Restored scheduler state")

        # Load uncertainty module if available
        if uncertainty_module and checkpoint.get("uncertainty_state_dict"):
            uncertainty_module. load_state_dict(checkpoint["uncertainty_state_dict"])
            logger.info("✅ Restored uncertainty module state")

        # Restore random states for reproducibility
        if "random_state" in checkpoint:
            torch.set_rng_state(checkpoint["random_state"]["torch_rng_state"])
            if checkpoint["random_state"]["cuda_rng_state"] and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(checkpoint["random_state"]["cuda_rng_state"])
            logger. info("✅ Restored random states")

        resume_info = {
            "start_epoch": checkpoint["epoch"] + 1,  # Resume from next epoch
            "last_metrics": checkpoint.get("metrics", {}),
            "extra_state": checkpoint.get("extra_state", {}),
        }

        logger.info(f"🔄 Resuming from epoch {checkpoint['epoch']}")
        logger.info(f"Last metrics: {resume_info['last_metrics']}")

        return resume_info

    def checkpoint_exists(self) -> bool:
        """Check if any checkpoint exists for this model."""
        return (self.save_dir / LATEST_CHECKPOINT_NAME).exists()

    def get_best_checkpoint_path(self) -> Optional[Path]:
        """Get path to best model checkpoint if it exists."""
        best_path = self.save_dir / BEST_MODEL_NAME
        return best_path if best_path. exists() else None

    def list_checkpoints(self) -> list[Path]:
        """List all checkpoint files for this model."""
        return sorted(self.save_dir.glob("checkpoint_epoch_*.pt"))

    def delete_all_checkpoints(self):
        """Delete all checkpoints for this model (use with caution!)."""
        if self.save_dir.exists():
            shutil.rmtree(self. save_dir)
            logger. warning(f"🗑️ Deleted all checkpoints in {self.save_dir}")
            self.save_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Private Helper Methods
    # -------------------------------------------------------------------------

    def _atomic_save(self, checkpoint: Dict, path: Path):
        """
        Atomically save checkpoint to prevent corruption.

        Saves to a temporary file first, then renames to avoid
        partial writes if process is interrupted.
        """
        temp_path = path. with_suffix(". tmp")
        try:
            torch.save(checkpoint, temp_path)
            temp_path.replace(path)  # Atomic on POSIX systems
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise ModelError(f"Failed to save checkpoint: {e}") from e

    def _save_checkpoint_info(self, epoch: int, metrics: Dict, is_best: bool):
        """Save human-readable checkpoint info as JSON."""
        info = {
            "model_name": self.model_name,
            "last_epoch": epoch,
            "last_metrics": metrics,
            "is_best": is_best,
            "timestamp": datetime.now().isoformat(),
            "checkpoint_dir": str(self.save_dir),
        }

        info_path = self.save_dir / CHECKPOINT_INFO_NAME
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)

    def _cleanup_old_checkpoints(self):
        """Remove old epoch checkpoints, keeping only the last N."""
        checkpoints = self.list_checkpoints()

        # Keep only the latest N checkpoints
        if len(checkpoints) > self.keep_last_n:
            to_delete = checkpoints[:  -self.keep_last_n]
            for ckpt in to_delete:
                ckpt. unlink()
                logger.debug(f"Deleted old checkpoint:  {ckpt. name}")


# =============================================================================
# Convenience Functions
# =============================================================================


def save_model_only(
    model: torch.nn.Module,
    model_name: str,
    save_path: Optional[Path] = None,
) -> Path:
    """
    Save only the model state dict (for deployment/inference).

    Args:
        model: PyTorch model.
        model_name: Name for the saved model.
        save_path: Custom save path (defaults to models/model_name. pt).

    Returns:
        Path to saved model.
    """
    if save_path is None:
        save_path = DIRS["models"] / f"{model_name}.pt"

    torch.save(model.state_dict(), save_path)
    logger.info(f"💾 Saved model to {save_path}")

    return save_path


def load_model_only(
    model: torch.nn.Module,
    model_path: Path,
    strict: bool = True,
) -> torch.nn.Module:
    """
    Load only model state dict (for inference).

    Args:
        model: Model instance to load weights into.
        model_path: Path to saved model file.
        strict: Whether to strictly enforce state dict keys match.

    Returns:
        Model with loaded weights.

    Raises:
        ModelError: If loading fails.
    """
    if not model_path.exists():
        raise ModelError(f"Model file not found: {model_path}")

    try:
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=strict)
        logger.info(f"✅ Loaded model from {model_path}")
        return model
    except Exception as e:
        raise ModelError(f"Failed to load model:  {e}") from e
