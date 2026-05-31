"""Shared bootstrap helpers for training, tuning, and inference engines.

The three engines (``StandardTrainer`` orchestrated by ``train_pipeline``,
``PGenTuner`` for Optuna, and ``PGenPredictor`` for inference) all need the
same plumbing: pick a device, load + clean the training data, split it,
build ``DoubleTowerDataset`` pairs, infer tower dimensions, assemble
``DataLoader``s, and instantiate the GNN model with the correct
``create_gnn_model`` signature. Pulling that out of each engine removes
~200 lines of duplication and keeps the architectural contract in one place.

Each helper is intentionally pure: no global state, no side effects beyond
filesystem access for data loading.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import polars as pl
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

from src.config import get_settings
from src.core import (
    ConfigurationError,
    ConfigValidator,
    DataError,
    DataValidator,
    ModelError,
)
from src.data.cleaning import PharmacogenomicCleaner
from src.data.collator import DoubleTowerCollater
from src.data.datasets import DoubleTowerDataset
from src.data.loaders import TabularLoader
from src.model.architectures.layers import create_gnn_model

logger = logging.getLogger(__name__)


PRELOAD_THRESHOLD = 10_000
DEFAULT_NUM_WORKERS = 4
MIN_DATASET_SIZE = 100


# --------------------------------------------------------------------------- #
# Device
# --------------------------------------------------------------------------- #


def resolve_device(override: str | None = None) -> torch.device:
    """Return a torch.device, preferring an explicit override.

    ``override`` accepts the same strings as ``torch.device`` ("cuda", "cpu",
    "cuda:0", …). When None, picks CUDA if available, else CPU.
    """
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Tower dimensions (legacy nested-dict spec consumed by DoubleTowerDataset)
# --------------------------------------------------------------------------- #


def extract_tower_dims(cfg) -> dict[str, dict[str, int]]:
    """Build the nested ``{tower: {features, edges, attrs}}`` dim spec.

    Values come from ``cfg.extras`` with sensible defaults. Each value must
    be a non-negative int — otherwise a ConfigurationError is raised.
    """
    extras = cfg.extras
    dims = {
        "drugs": {
            "features": extras.get("drug_in_features", 25),
            "edges": extras.get("drug_edge_dim", 7),
            "attrs": extras.get("drug_in_attributes", 0),
        },
        "geno": {
            "features": extras.get("haplo_in_features", 9),
            "edges": extras.get("haplo_edge_dim", 3),
            "attrs": extras.get("haplo_in_attributes", 0),
        },
    }
    for tower, sub in dims.items():
        for name, value in sub.items():
            if not isinstance(value, int) or value < 0:
                raise ConfigurationError(
                    f"Invalid dimension '{tower}.{name}': {value} (must be >= 0 integer)"
                )
    return dims


# --------------------------------------------------------------------------- #
# Data loading + validation
# --------------------------------------------------------------------------- #


def load_and_clean_data(
    csv_path: str | Path,
    cfg,
    *,
    enforce_min_size: int = MIN_DATASET_SIZE,
) -> pl.DataFrame:
    """Load a CSV/TSV through TabularLoader and run PharmacogenomicCleaner.

    Validates the result against the configured features/targets and warns
    on high missingness. Raises DataError on any failure that would prevent
    a meaningful training run.
    """
    try:
        raw_df = TabularLoader.load(csv_path, columns=cfg.cols or None)
        df = PharmacogenomicCleaner(
            multi_label_cols=get_settings().multi_label_set,
        ).clean(raw_df, stratify_col=cfg.stratify_col)
    except Exception as e:
        raise DataError(f"Failed to load data from {csv_path}: {e}") from e

    if len(df) < enforce_min_size:
        raise DataError(
            f"Dataset too small: {len(df)} samples (minimum: {enforce_min_size})"
        )

    try:
        ConfigValidator.validate_data_columns(df.columns, cfg.features, cfg.targets)
    except ValueError as e:
        raise DataError(str(e)) from e

    missing_stats = DataValidator.check_missing_values(
        df, cfg.features + cfg.targets, threshold=0.5
    )
    if any(frac > 0.1 for frac in missing_stats.values()):
        logger.warning("High missing values detected: %s", missing_stats)

    return df


# --------------------------------------------------------------------------- #
# Train / val split
# --------------------------------------------------------------------------- #


def stratified_split(
    df: pl.DataFrame,
    validation_split: float,
    *,
    seed: int | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split ``df`` into (train, val), stratifying on ``_stratify`` if present."""
    seed = seed if seed is not None else get_settings().seed
    stratify = df["_stratify"] if "_stratify" in df.columns else None
    train_df, val_df = train_test_split(
        df, test_size=validation_split, stratify=stratify, random_state=seed
    )
    return train_df, val_df


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #


def build_two_tower_datasets(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    cfg,
    dims: dict[str, dict[str, int]],
    *,
    preload_ram: bool | None = None,
) -> tuple[DoubleTowerDataset, DoubleTowerDataset]:
    """Construct paired train/val DoubleTowerDatasets.

    The val dataset reuses the train dataset's fitted target encoders so
    label spaces stay aligned.
    """
    if preload_ram is None:
        preload_ram = len(train_df) < PRELOAD_THRESHOLD

    multi_label_cols = list(get_settings().multi_label_set)

    train_dataset = DoubleTowerDataset(
        df=train_df,
        drug_col=cfg.features[0],
        geno_col=cfg.features[1],
        target_cols=cfg.targets,
        multilabel_cols=multi_label_cols,
        preload_ram=preload_ram,
        input_dimensions=dims,
    )

    val_dataset = DoubleTowerDataset(
        df=val_df,
        drug_col=cfg.features[0],
        geno_col=cfg.features[1],
        target_cols=cfg.targets,
        multilabel_cols=multi_label_cols,
        encoders=train_dataset.target_encoder.encoders,
        preload_ram=preload_ram,
        input_dimensions=dims,
    )

    return train_dataset, val_dataset


def infer_dataset_dimensions(
    train_dataset: DoubleTowerDataset,
    cfg,
) -> tuple[int, int, dict[str, int]]:
    """Probe a sample to learn the per-tower feature widths and per-target sizes."""
    try:
        sample = train_dataset[0]
        drug_dim = sample["drug_data"].x.shape[1]
        geno_dim = sample["geno_data"].x.shape[1]
    except Exception as e:
        raise DataError(f"Failed to infer dimensions from dataset: {e}") from e

    if drug_dim <= 0 or geno_dim <= 0:
        raise DataError(
            f"Invalid inferred dimensions: drug={drug_dim}, geno={geno_dim}"
        )

    encoders = train_dataset.target_encoder.encoders
    target_dims: dict[str, int] = {}
    for col in cfg.targets:
        if col not in encoders:
            raise DataError(f"Target column '{col}' not encoded")
        target_dims[col] = len(encoders[col].classes_)

    return drug_dim, geno_dim, target_dims


# --------------------------------------------------------------------------- #
# DataLoaders
# --------------------------------------------------------------------------- #


def build_train_val_loaders(
    train_dataset: DoubleTowerDataset,
    val_dataset: DoubleTowerDataset,
    batch_size: int,
    *,
    num_workers: int | None = None,
    collater: DoubleTowerCollater | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Build paired train + val DataLoaders with the project's standard knobs."""
    collater = collater or DoubleTowerCollater()
    if num_workers is None:
        preload_ram = len(train_dataset) < PRELOAD_THRESHOLD
        cpu_count = os.cpu_count() or 1
        num_workers = (
            2 if preload_ram else min(DEFAULT_NUM_WORKERS, max(cpu_count - 1, 1))
        )

    pin = torch.cuda.is_available()
    persistent = num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collater,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collater,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=persistent,
    )
    return train_loader, val_loader


# --------------------------------------------------------------------------- #
# Model construction
# --------------------------------------------------------------------------- #


def build_gnn_model(
    *,
    model_name: str,
    dims: dict[str, dict[str, int]],
    drug_dim: int,
    geno_dim: int,
    target_dims: dict[str, int],
    params: dict[str, Any],
    device: torch.device,
) -> nn.Module:
    """Instantiate PharmagenTwoTower with the inferred / configured shapes."""
    drug_config = {"num_features": drug_dim, "edge_dim": dims["drugs"]["edges"]}
    geno_config = {"num_features": geno_dim, "edge_dim": dims["geno"]["edges"]}
    try:
        model = create_gnn_model(
            model_name=model_name,
            drug_config=drug_config,
            geno_config=geno_config,
            target_dims=target_dims,
            params=params,
        ).to(device)
    except Exception as e:
        raise ModelError(f"Failed to create model '{model_name}': {e}") from e
    return model


__all__ = [
    "DEFAULT_NUM_WORKERS",
    "MIN_DATASET_SIZE",
    "PRELOAD_THRESHOLD",
    "build_gnn_model",
    "build_train_val_loaders",
    "build_two_tower_datasets",
    "extract_tower_dims",
    "infer_dataset_dimensions",
    "load_and_clean_data",
    "resolve_device",
    "stratified_split",
]
