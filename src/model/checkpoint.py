"""Training checkpoint management.

:class:`CheckpointManager` handles atomic saves, best-model tracking,
and rolling cleanup of old epoch checkpoints. :func:`save_model_only` /
:func:`load_model_only` provide lightweight helpers for inference-only use.
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from src.config import get_settings
from src.core import ModelError

logger = logging.getLogger(__name__)

BEST_MODEL_NAME = "best_model.pt"
LATEST_CHECKPOINT_NAME = "latest_checkpoint.pt"
CHECKPOINT_INFO_NAME = "checkpoint_info.json"


class CheckpointManager:
    """Manages training checkpoints with atomic saving and automatic cleanup."""

    def __init__(
        self,
        model_name: str,
        save_dir: Path | None = None,
        keep_last_n: int = 3,
    ):
        self.model_name = model_name
        checkpoint_base = get_settings().paths.models / "checkpoints"
        self.save_dir = save_dir or (checkpoint_base / model_name)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        logger.debug(
            "CheckpointManager ready (name=%s, dir=%s)", model_name, self.save_dir
        )

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: dict[str, float],
        scheduler: Any | None = None,
        uncertainty_module: torch.nn.Module | None = None,
        is_best: bool = False,
        extra_state: dict[str, Any] | None = None,
    ) -> Path:
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "uncertainty_state_dict": (
                uncertainty_module.state_dict() if uncertainty_module else None
            ),
            "model_name": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "random_state": {
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                ),
            },
            "extra_state": extra_state or {},
        }

        latest_path = self.save_dir / LATEST_CHECKPOINT_NAME
        self._atomic_save(checkpoint, latest_path)
        logger.info("Saved checkpoint at epoch %d to %s", epoch, latest_path)

        epoch_path = self.save_dir / f"checkpoint_epoch_{epoch:04d}.pt"
        self._atomic_save(checkpoint, epoch_path)
        logger.debug("Saved epoch checkpoint: %s", epoch_path)

        if is_best:
            best_path = self.save_dir / BEST_MODEL_NAME
            self._atomic_save(checkpoint, best_path)
            logger.info("New best model saved with metrics: %s", metrics)

        self._save_checkpoint_info(epoch, metrics, is_best)

        if self.keep_last_n > 0:
            self._cleanup_old_checkpoints()

        return latest_path

    def load_checkpoint(
        self,
        checkpoint_path: Path | None = None,
        load_best: bool = False,
    ) -> dict[str, Any]:
        if checkpoint_path:
            path = Path(checkpoint_path)
        elif load_best:
            path = self.save_dir / BEST_MODEL_NAME
        else:
            path = self.save_dir / LATEST_CHECKPOINT_NAME

        if not path.exists():
            raise ModelError(f"Checkpoint not found: {path}")

        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            logger.info(
                "Loaded checkpoint from %s (epoch %s)",
                path,
                checkpoint.get("epoch", "?"),
            )
            return checkpoint
        except Exception as e:
            raise ModelError(f"Failed to load checkpoint from {path}: {e}") from e

    def resume_training(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any | None = None,
        uncertainty_module: torch.nn.Module | None = None,
        checkpoint_path: Path | None = None,
    ) -> dict[str, Any]:
        checkpoint = self.load_checkpoint(checkpoint_path, load_best=True)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler and checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if uncertainty_module and checkpoint.get("uncertainty_state_dict"):
            uncertainty_module.load_state_dict(checkpoint["uncertainty_state_dict"])

        if "random_state" in checkpoint:
            torch.set_rng_state(checkpoint["random_state"]["torch_rng_state"])
            cuda_state = checkpoint["random_state"]["cuda_rng_state"]
            if cuda_state and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(cuda_state)

        return {
            "start_epoch": checkpoint["epoch"] + 1,
            "last_metrics": checkpoint.get("metrics", {}),
            "extra_state": checkpoint.get("extra_state", {}),
        }

    def checkpoint_exists(self) -> bool:
        return (self.save_dir / LATEST_CHECKPOINT_NAME).exists()

    def get_best_checkpoint_path(self) -> Path | None:
        best_path = self.save_dir / BEST_MODEL_NAME
        return best_path if best_path.exists() else None

    def list_checkpoints(self) -> list[Path]:
        return sorted(self.save_dir.glob("checkpoint_epoch_*.pt"))

    def delete_all_checkpoints(self) -> None:
        if self.save_dir.exists():
            shutil.rmtree(self.save_dir)
            logger.warning("Deleted all checkpoints in %s", self.save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def _atomic_save(self, checkpoint: dict, path: Path) -> None:
        temp = path.with_suffix(".tmp")
        try:
            torch.save(checkpoint, temp)
            temp.replace(path)
        except Exception as e:
            if temp.exists():
                temp.unlink()
            raise ModelError(f"Failed to save checkpoint: {e}") from e

    def _save_checkpoint_info(self, epoch: int, metrics: dict, is_best: bool) -> None:
        info = {
            "model_name": self.model_name,
            "last_epoch": epoch,
            "last_metrics": metrics,
            "is_best": is_best,
            "timestamp": datetime.now().isoformat(),
            "checkpoint_dir": str(self.save_dir),
        }
        with open(self.save_dir / CHECKPOINT_INFO_NAME, "w") as f:
            json.dump(info, f, indent=2)

    def _cleanup_old_checkpoints(self) -> None:
        checkpoints = self.list_checkpoints()
        for ckpt in checkpoints[: -self.keep_last_n]:
            ckpt.unlink()
            logger.debug("Deleted old checkpoint: %s", ckpt.name)


def save_model_only(
    model: torch.nn.Module,
    model_name: str,
    save_path: Path | None = None,
) -> Path:
    if save_path is None:
        save_path = get_settings().paths.models / f"{model_name}.pt"
    torch.save(model.state_dict(), save_path)
    logger.info("Saved model weights to %s", save_path)
    return save_path


def load_model_only(
    model: torch.nn.Module,
    model_path: Path,
    strict: bool = True,
) -> torch.nn.Module:
    if not model_path.exists():
        raise ModelError(f"Model file not found: {model_path}")
    try:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=strict)
        logger.info("Loaded model weights from %s", model_path)
        return model
    except Exception as e:
        raise ModelError(f"Failed to load model: {e}") from e
