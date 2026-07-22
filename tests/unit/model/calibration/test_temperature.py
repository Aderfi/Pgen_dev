import torch

from src.model.architectures.config import TaskSpec
from src.model.calibration import TemperatureScaler


def test_forward_scales_logits_by_temperature():
    specs = {"t": TaskSpec(dim=1, kind="binary")}
    scaler = TemperatureScaler(specs)
    outputs = {"t": torch.randn(5, 1)}
    scaled = scaler(outputs)
    assert "t" in scaled
    assert scaled["t"].shape == outputs["t"].shape
    # log_temp starts at 0, so exp(0) == 1: forward is a no-op initially.
    assert torch.allclose(scaled["t"], outputs["t"])


def test_forward_skips_tasks_not_in_outputs():
    specs = {"t": TaskSpec(dim=1, kind="binary")}
    scaler = TemperatureScaler(specs)
    scaled = scaler({})
    assert scaled == {}


def test_fit_updates_log_temp_parameter():
    specs = {"t": TaskSpec(dim=1, kind="binary")}
    scaler = TemperatureScaler(specs)
    before = scaler.log_temp["t"].clone()

    logits = {"t": torch.randn(20, 1) * 5}
    labels = {"t": (torch.rand(20, 1) > 0.5).float()}
    scaler.fit(logits, labels, specs, max_iter=5)

    after = scaler.log_temp["t"]
    assert not torch.equal(before, after)
