"""Training checkpoint management.

:class:`CheckpointManager` handles atomic saves, best-model tracking,
and rolling cleanup of old epoch checkpoints.

Checkpoints are written as safetensors files. A checkpoint is a nested
structure (model / optimizer / scheduler / uncertainty state dicts, RNG
states, epoch, metrics), but safetensors only stores a flat ``name -> tensor``
map plus ``str -> str`` metadata. So every tensor is stored under its path in
the structure (``model_state_dict/encoder.weight``) and the rest of the
structure goes, as JSON, into the ``pharmagen_checkpoint`` metadata entry with
``{"__tensor__": name}`` placeholders where the tensors were. Loading never
unpickles anything.

Legacy ``.pt`` (``torch.save``) checkpoints are still read when no
safetensors file exists next to them.
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from src.config import get_settings
from src.core import ModelError

logger = logging.getLogger(__name__)

CHECKPOINT_SUFFIX = ".safetensors"
LEGACY_SUFFIX = ".pt"
BEST_MODEL_NAME = f"best_model{CHECKPOINT_SUFFIX}"
LATEST_CHECKPOINT_NAME = f"latest_checkpoint{CHECKPOINT_SUFFIX}"
CHECKPOINT_INFO_NAME = "checkpoint_info.json"

METADATA_KEY = "pharmagen_checkpoint"
_TENSOR = "__tensor__"
_TUPLE = "__tuple__"
_DICT = "__dict__"
_MARKERS = frozenset({_TENSOR, _TUPLE, _DICT})


def _join(path: str, key: object) -> str:
    return f"{path}/{key}" if path else str(key)


def _encode(obj: Any, path: str, tensors: dict[str, torch.Tensor]) -> Any:
    """JSON-safe copy of ``obj``; its tensors are moved into ``tensors``."""
    if isinstance(obj, torch.Tensor):
        if obj.layout != torch.strided:
            raise ModelError(
                f"Sparse tensor at {path!r} ({obj.layout}): safetensors only "
                "stores dense tensors; keep edge_index / edge_weight dense instead"
            )
        if path in tensors:
            raise ModelError(f"Duplicate tensor name in checkpoint: {path!r}")
        # Own CPU copy: safetensors refuses tensors that share storage.
        tensors[path] = obj.detach().to("cpu", copy=True).contiguous()
        return {_TENSOR: path}
    if obj is None or isinstance(obj, bool | int | float | str):
        return obj
    if isinstance(obj, tuple):
        return {
            _TUPLE: [_encode(v, _join(path, i), tensors) for i, v in enumerate(obj)]
        }
    if isinstance(obj, list):
        return [_encode(v, _join(path, i), tensors) for i, v in enumerate(obj)]
    if isinstance(obj, dict):
        if all(isinstance(k, str) for k in obj) and not _MARKERS & obj.keys():
            return {k: _encode(v, _join(path, k), tensors) for k, v in obj.items()}
        # Non-string keys (optimizer state is keyed by param index) keep their type.
        return {
            _DICT: [
                [
                    _encode(k, _join(path, f"key{i}"), tensors),
                    _encode(v, _join(path, k), tensors),
                ]
                for i, (k, v) in enumerate(obj.items())
            ]
        }
    raise ModelError(
        f"Cannot store {type(obj).__name__} at {path!r} in a safetensors checkpoint"
    )


def _decode(obj: Any, tensors: dict[str, torch.Tensor]) -> Any:
    """Inverse of :func:`_encode`."""
    if isinstance(obj, list):
        return [_decode(v, tensors) for v in obj]
    if not isinstance(obj, dict):
        return obj
    if len(obj) == 1:
        ((key, value),) = obj.items()
        if key == _TENSOR:
            return tensors[value]
        if key == _TUPLE:
            return tuple(_decode(v, tensors) for v in value)
        if key == _DICT:
            return {_decode(k, tensors): _decode(v, tensors) for k, v in value}
    return {k: _decode(v, tensors) for k, v in obj.items()}


def save_checkpoint_file(checkpoint: dict[str, Any], path: Path) -> None:
    """Write ``checkpoint`` (any nesting of tensors + JSON values) to ``path``."""
    tensors: dict[str, torch.Tensor] = {}
    structure = _encode(checkpoint, "", tensors)
    save_file(
        tensors,
        str(path),
        metadata={"format": "pt", METADATA_KEY: json.dumps(structure)},
    )


def load_checkpoint_file(path: Path) -> dict[str, Any]:
    """Read a checkpoint written by :func:`save_checkpoint_file` (CPU tensors)."""
    with safe_open(str(path), framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
        tensors = {name: f.get_tensor(name) for name in f.keys()}
    if METADATA_KEY not in metadata:
        raise ModelError(f"{path} is not a Pharmagen checkpoint (no {METADATA_KEY!r})")
    return _decode(json.loads(metadata[METADATA_KEY]), tensors)


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

        epoch_path = self.save_dir / f"checkpoint_epoch_{epoch:04d}{CHECKPOINT_SUFFIX}"
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
            path = self._resolve(BEST_MODEL_NAME)
        else:
            path = self._resolve(LATEST_CHECKPOINT_NAME)

        if not path.exists():
            raise ModelError(f"Checkpoint not found: {path}")

        try:
            if path.suffix == LEGACY_SUFFIX:
                logger.warning(
                    "Loading legacy torch.save checkpoint %s; retrain to write "
                    "a safetensors checkpoint.",
                    path,
                )
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            else:
                checkpoint = load_checkpoint_file(path)
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
        return self._resolve(LATEST_CHECKPOINT_NAME).exists()

    def get_best_checkpoint_path(self) -> Path | None:
        best_path = self._resolve(BEST_MODEL_NAME)
        return best_path if best_path.exists() else None

    def list_checkpoints(self) -> list[Path]:
        return sorted(self.save_dir.glob(f"checkpoint_epoch_*{CHECKPOINT_SUFFIX}"))

    def delete_all_checkpoints(self) -> None:
        if self.save_dir.exists():
            shutil.rmtree(self.save_dir)
            logger.warning("Deleted all checkpoints in %s", self.save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def _resolve(self, name: str) -> Path:
        """``save_dir / name``, or its legacy ``.pt`` twin if only that exists."""
        path = self.save_dir / name
        legacy = path.with_suffix(LEGACY_SUFFIX)
        if not path.exists() and legacy.exists():
            return legacy
        return path

    def _atomic_save(self, checkpoint: dict, path: Path) -> None:
        temp = path.with_suffix(".tmp")
        try:
            save_checkpoint_file(checkpoint, temp)
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
