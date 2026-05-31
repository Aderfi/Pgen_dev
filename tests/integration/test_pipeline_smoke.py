"""Smoke tests that exercise top-level imports and helpers.

These tests stay cheap on purpose: they protect against regressions like
the historical ``MemoryMonitor`` NameError, where ``train_pipeline`` would
crash at runtime because a helper referenced an undefined symbol. Heavy
end-to-end training is intentionally out of scope here.
"""

from __future__ import annotations

import importlib

import pytest

pytestmark = pytest.mark.integration


def test_train_pipeline_imports_cleanly():
    module = importlib.import_module("src.pipeline")
    assert hasattr(module, "train_pipeline"), "train_pipeline missing from src.pipeline"


def test_pipeline_local_helpers_are_defined():
    """Local pipeline helpers (the ones not yet pushed into engine.base)."""
    module = importlib.import_module("src.pipeline")
    for helper in (
        "_announce_device",
        "_log_memory_stats",
        "_persist_training_artifacts",
        "_setup_trainer",
        "_execute_training",
    ):
        assert hasattr(module, helper), f"pipeline helper {helper!r} missing"


def test_engine_base_exposes_shared_helpers():
    base = importlib.import_module("src.model.engine.base")
    for sym in (
        "resolve_device",
        "extract_tower_dims",
        "load_and_clean_data",
        "stratified_split",
        "build_two_tower_datasets",
        "infer_dataset_dimensions",
        "build_train_val_loaders",
        "build_gnn_model",
    ):
        assert hasattr(base, sym), f"engine.base helper {sym!r} missing"


def test_resolve_device_returns_torch_device():
    import torch

    from src.model.engine.base import resolve_device

    assert isinstance(resolve_device(), torch.device)
    assert resolve_device("cpu").type == "cpu"


def test_log_memory_stats_is_safe_without_cuda():
    module = importlib.import_module("src.pipeline")
    module._log_memory_stats("smoke")


def test_engine_modules_import():
    importlib.import_module("src.model.engine.predictor")
    importlib.import_module("src.model.engine.tuner")
    importlib.import_module("src.model.engine.base")


def test_predictor_requires_encoders_to_instantiate():
    """Without trained artifacts, PGenPredictor must fail loudly, not silently."""
    from src.model.engine.predictor import PGenPredictor

    with pytest.raises(FileNotFoundError, match="[Ee]ncoders file not found"):
        PGenPredictor("TwoTowerGAT")


def test_api_app_imports():
    api = importlib.import_module("src.api.main")
    assert hasattr(api, "app"), "FastAPI app missing in src.api.main"


def test_cli_entry_imports():
    importlib.import_module("src.interface.cli")


def test_settings_paths_resolves():
    from src.config import get_settings

    paths = get_settings().paths
    assert paths.project_root.exists()
    assert paths.data.is_absolute()
    assert paths.library.is_absolute()
