import torch

from src.model.architectures.config import TaskSpec
from src.model.losses import MultiTaskLoss, focal_bce_with_logits


def test_focal_reduces_to_bce_when_gamma_zero():
    logits = torch.randn(10, 1)
    y = (torch.rand(10, 1) > 0.5).float()
    focal = focal_bce_with_logits(logits, y, gamma=0.0)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
    assert torch.allclose(focal, bce, atol=1e-6)


def test_multitask_masks_nan_targets():
    specs = {"t": TaskSpec(dim=1, kind="binary")}
    loss = MultiTaskLoss(specs)
    outputs = {"t": torch.randn(4, 1)}
    y = torch.tensor([[1.0], [float("nan")], [0.0], [1.0]])
    total, per_task = loss(outputs, {"t": y})
    assert torch.isfinite(total)
    assert "t" in per_task
