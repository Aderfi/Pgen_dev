import pytest
import torch
from torch import nn

from src.config.models import TrainingParams
from src.model.factories import build_optimizer, build_scheduler

BASE = {
    "embedding_dim": 16,
    "hidden_dim": 16,
    "n_layers": 2,
    "heads": 2,
    "dropout_rate": 0.1,
}


@pytest.mark.parametrize(
    ("optimizer_type", "expected_cls"),
    [
        ("adamw", torch.optim.AdamW),
        ("adam", torch.optim.Adam),
        ("sgd", torch.optim.SGD),
        ("rmsprop", torch.optim.RMSprop),
    ],
)
def test_build_optimizer_dispatches_on_type(optimizer_type, expected_cls):
    model = nn.Linear(4, 2)
    params = TrainingParams(**BASE, optimizer_type=optimizer_type)
    optimizer = build_optimizer(model, params)
    assert isinstance(optimizer, expected_cls)


def test_build_optimizer_rejects_unknown_type():
    model = nn.Linear(4, 2)
    params = TrainingParams(**BASE)
    params.optimizer_type = (
        "made-up"  # bypass Literal validation to hit the else-branch
    )
    with pytest.raises(ValueError, match="Unknown optimizer_type"):
        build_optimizer(model, params)


def test_build_optimizer_adds_uncertainty_param_group_without_weight_decay():
    model = nn.Linear(4, 2)
    uncertainty = nn.Linear(1, 1)
    params = TrainingParams(**BASE, learning_rate=1e-2, loss_learning_rate=5e-2)
    optimizer = build_optimizer(model, params, uncertainty_module=uncertainty)
    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[1]["weight_decay"] == 0.0
    assert optimizer.param_groups[1]["lr"] == 5e-2


def test_build_scheduler_reads_factor_and_patience_from_params():
    model = nn.Linear(4, 2)
    params = TrainingParams(**BASE, scheduler_factor=0.25, scheduler_patience=3)
    optimizer = build_optimizer(model, params)
    scheduler = build_scheduler(optimizer, params)
    assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    assert scheduler.factor == 0.25
    assert scheduler.patience == 3
