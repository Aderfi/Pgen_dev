"""Safetensors checkpoint round-trip for :class:`CheckpointManager`."""

from typing import TYPE_CHECKING

import pytest
import torch
from safetensors import safe_open

from src.core import ModelError
from src.model.checkpoint import (
    BEST_MODEL_NAME,
    CheckpointManager,
    load_checkpoint_file,
    save_checkpoint_file,
)

if TYPE_CHECKING:
    from pathlib import Path


class _Tied(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = torch.nn.Linear(4, 4)
        self.b = torch.nn.Linear(4, 4)
        self.b.weight = self.a.weight  # shared storage
        self.register_buffer("count", torch.tensor(3, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(self.a(x))


def _trained(optimizer_cls=torch.optim.AdamW):
    torch.manual_seed(0)
    model = _Tied()
    uncertainty = torch.nn.Linear(2, 1)
    opt = optimizer_cls(
        [
            {"params": model.parameters()},
            {"params": uncertainty.parameters(), "weight_decay": 0.0},
        ],
        lr=1e-2,
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=1)
    for _ in range(3):
        opt.zero_grad()
        loss = model(torch.randn(5, 4)).sum() + uncertainty(torch.randn(3, 2)).sum()
        loss.backward()
        opt.step()
        sched.step(loss.item())
    return model, uncertainty, opt, sched


def _assert_same(a, b) -> None:
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.dtype == b.dtype
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            _assert_same(a[k], b[k])
    elif isinstance(a, list | tuple):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            _assert_same(x, y)
    else:
        assert a == b


def test_file_roundtrip_preserves_structure(tmp_path: Path) -> None:
    model, unc, opt, sched = _trained()
    state = {
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "sched": sched.state_dict(),
        "unc": unc.state_dict(),
        "misc": {"inf": float("inf"), "none": None, "tup": (1, "x"), 7: "int-key"},
    }
    path = tmp_path / "ckpt.safetensors"
    save_checkpoint_file(state, path)
    _assert_same(state, load_checkpoint_file(path))


def test_file_is_plain_safetensors(tmp_path: Path) -> None:
    model, *_ = _trained()
    path = tmp_path / "ckpt.safetensors"
    save_checkpoint_file({"model_state_dict": model.state_dict()}, path)
    with safe_open(str(path), framework="pt") as f:
        assert "model_state_dict/a.weight" in set(f.keys())


def test_unsupported_value_raises(tmp_path: Path) -> None:
    with pytest.raises(ModelError, match="Cannot store"):
        save_checkpoint_file({"bad": object()}, tmp_path / "x.safetensors")


def test_manager_save_and_resume(tmp_path: Path) -> None:
    model, unc, opt, sched = _trained()
    manager = CheckpointManager("m", save_dir=tmp_path, keep_last_n=2)
    for epoch in range(4):
        manager.save_checkpoint(
            model, opt, epoch, {"val_loss": 1.0 / (epoch + 1)}, sched, unc, is_best=True
        )
    assert manager.get_best_checkpoint_path() == tmp_path / BEST_MODEL_NAME
    assert [p.name for p in manager.list_checkpoints()] == [
        "checkpoint_epoch_0002.safetensors",
        "checkpoint_epoch_0003.safetensors",
    ]
    assert not list(tmp_path.glob("*.tmp"))

    fresh, fresh_unc, fresh_opt, fresh_sched = _trained(torch.optim.AdamW)
    with torch.no_grad():
        for p in fresh.parameters():
            p.zero_()
    info = manager.resume_training(fresh, fresh_opt, fresh_sched, fresh_unc)

    assert info["start_epoch"] == 4
    assert info["last_metrics"] == {"val_loss": 0.25}
    _assert_same(model.state_dict(), fresh.state_dict())
    _assert_same(opt.state_dict(), fresh_opt.state_dict())
    _assert_same(sched.state_dict(), fresh_sched.state_dict())
    _assert_same(unc.state_dict(), fresh_unc.state_dict())


def test_legacy_pt_checkpoint_still_loads(tmp_path: Path) -> None:
    model, *_ = _trained()
    torch.save(
        {"model_state_dict": model.state_dict(), "epoch": 5},
        tmp_path / "best_model.pt",
    )
    manager = CheckpointManager("m", save_dir=tmp_path)
    assert manager.get_best_checkpoint_path() == tmp_path / "best_model.pt"
    ckpt = manager.load_checkpoint(load_best=True)
    assert ckpt["epoch"] == 5
    _assert_same(model.state_dict(), ckpt["model_state_dict"])


def test_sparse_tensor_raises(tmp_path: Path) -> None:
    sparse = torch.eye(3).to_sparse()
    with pytest.raises(ModelError, match="Sparse tensor"):
        save_checkpoint_file({"adj": sparse}, tmp_path / "x.safetensors")
