# src/pipeline.py
# Pharmagen - Training Pipeline

"""Standard training: data → model → trainer → checkpoints + artifact bundle.

All bootstrap lives in :mod:`src.model.engine.base`; this module only
sequences it and persists what inference needs (:mod:`src.model.artifacts`).
"""

from __future__ import annotations

import gc
import logging
from typing import TYPE_CHECKING

import torch

from src.config import get_model_config
from src.core import ConfigurationError
from src.evaluation.runs import run_name
from src.model.artifacts import save_training_artifacts
from src.model.engine.base import (
    build_gnn_model,
    build_train_val_loaders,
    build_trainer,
    model_switches,
    prepare_training_data,
    resolve_device,
)

if TYPE_CHECKING:
    from src.data.datasets import DoubleTowerDataset
    from src.evaluation.runs import FoldRef
    from src.model.engine.base import DataSource

logger = logging.getLogger(__name__)

MIN_VALIDATION_SPLIT = 0.1
MAX_VALIDATION_SPLIT = 0.3


def train_pipeline(
    model_name: str,
    source: DataSource,
    epochs: int = 50,
    batch_size: int | None = None,
    validation_split: float = 0.2,
    patience: int = 10,
    split: FoldRef | None = None,
) -> float:
    """Train ``model_name`` on ``source`` (a TSV/CSV/parquet path, or ``"neo4j"``).

    ``batch_size`` defaults to the model's ``params.batch_size``. With
    ``split`` the model trains on that scenario fold and is saved under its
    run name (``src.evaluation.runs.run_name``). Returns the best validation
    loss.

    Raises:
        ConfigurationError: Invalid model configuration or split.
        DataError: Missing, invalid or incompatible data.
        ModelError: Model construction failed.
    """
    if not MIN_VALIDATION_SPLIT <= validation_split <= MAX_VALIDATION_SPLIT:
        raise ConfigurationError(
            f"validation_split must be in [{MIN_VALIDATION_SPLIT}, "
            f"{MAX_VALIDATION_SPLIT}], got {validation_split}"
        )
    try:
        cfg = get_model_config(model_name)
        params = cfg.training_params()
    except Exception as e:
        raise ConfigurationError(
            f"Failed to load configuration for '{model_name}': {e}"
        ) from e

    device = resolve_device()
    logger.info("Training %s on %s", model_name, device)
    _log_memory_stats("initial")

    data = prepare_training_data(
        cfg, source, validation_split=validation_split, split=split
    )
    name = run_name(model_name, split)
    switches = model_switches(cfg)
    model = build_gnn_model(
        dims=data.dims,
        drug_dim=data.drug_dim,
        geno_dim=data.geno_dim,
        axes=data.axes,
        params=params,
        device=device,
        switches=switches,
        gene_vocab=data.gene_vocab,
    )
    logger.info("Model parameters: %d", sum(p.numel() for p in model.parameters()))

    label_tuples, label_names = _build_label_table(
        data.train, model.axis_heads.single_label_axes()
    )
    logger.info("Compositional label table: %d unique tuples", len(label_tuples))
    save_training_artifacts(
        name,
        encoders=data.train.target_encoder.encoders,
        drug_dim=data.drug_dim,
        geno_dim=data.geno_dim,
        dims=data.dims,
        axes=data.axes,
        switches=switches,
        gene_vocab=data.gene_vocab,
        label_table={"tuples": [list(t) for t in label_tuples], "labels": label_names},
        split_info=(
            {
                "scenario": str(split.scenario),
                "fold": split.fold,
                "split_sha256": data.split_sha256 or "",
            }
            if split is not None
            else None
        ),
    )

    train_loader, val_loader = build_train_val_loaders(
        data.train, data.val, batch_size or params.batch_size
    )
    trainer = build_trainer(model, cfg, data.axes, params, device, checkpoint_name=name)
    try:
        return trainer.fit(train_loader, val_loader, epochs=epochs, patience=patience)
    except torch.OutOfMemoryError:
        logger.error("Out of memory during training; try a smaller batch size")
        raise
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _log_memory_stats(stage: str) -> None:
    if not torch.cuda.is_available():
        return
    logger.info(
        "GPU %s (%s): allocated=%.2f GB reserved=%.2f GB",
        torch.cuda.get_device_name(0),
        stage,
        torch.cuda.memory_allocated() / 1024**3,
        torch.cuda.memory_reserved() / 1024**3,
    )


def _build_label_table(
    train_dataset: DoubleTowerDataset, composable: list[str]
) -> tuple[list[tuple[int, ...]], list[str]]:
    """Unique training tuples over the composable axes, with readable labels.

    Each distinct combination of class indices across ``composable`` axes in
    ``train_dataset.targets`` gets the label ``"axis=value|axis2=value2"``
    (values inverse-transformed through the fitted encoders). Returns parallel
    ``(tuples, labels)`` lists, empty when ``composable`` is empty.
    """
    if not composable:
        return [], []

    encoders = train_dataset.target_encoder.encoders
    targets = train_dataset.targets
    n_rows = len(targets[composable[0]])

    seen: dict[tuple[int, ...], None] = {}
    for i in range(n_rows):
        seen.setdefault(tuple(int(targets[axis][i]) for axis in composable), None)

    tuples = list(seen)
    labels = [
        "|".join(
            f"{axis}={encoders[axis].inverse_transform([idx])[0]}"
            for axis, idx in zip(composable, row)
        )
        for row in tuples
    ]
    return tuples, labels


__all__ = ["train_pipeline"]
