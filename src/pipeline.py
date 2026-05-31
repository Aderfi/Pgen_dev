# src/pipeline.py
# Pharmagen - Training Pipeline
# Orchestrates data loading, model setup, and training loop.

from __future__ import annotations

import gc
import logging
from pathlib import Path

import joblib
import torch

from src.config import get_model_config, get_settings
from src.core import ConfigurationError
from src.interface.ui import ConsoleIO
from src.model.engine.base import (
    build_gnn_model,
    build_train_val_loaders,
    build_two_tower_datasets,
    extract_tower_dims,
    infer_dataset_dimensions,
    load_and_clean_data,
    resolve_device,
    stratified_split,
)
from src.model.factories import LossFactory, OptimizerFactory
from src.model.training.standard import StandardTrainer

logger = logging.getLogger(__name__)

MIN_VALIDATION_SPLIT = 0.1
MAX_VALIDATION_SPLIT = 0.3


def train_pipeline(
    model_name: str,
    csv_path: str | Path,
    epochs: int = 50,
    batch_size: int = 64,
    validation_split: float = 0.2,
    patience: int = 10,
) -> None:
    """Standard training pipeline with comprehensive validation and monitoring.

    Raises:
        ConfigurationError: If model configuration is invalid.
        DataError: If data is invalid or incompatible.
        ModelError: If model creation fails.
    """
    if not MIN_VALIDATION_SPLIT <= validation_split <= MAX_VALIDATION_SPLIT:
        raise ConfigurationError(
            f"validation_split must be in [{MIN_VALIDATION_SPLIT}, "
            f"{MAX_VALIDATION_SPLIT}], got {validation_split}"
        )

    try:
        cfg = get_model_config(model_name)
    except Exception as e:
        raise ConfigurationError(
            f"Failed to load configuration for '{model_name}': {e}"
        ) from e

    device = resolve_device()
    logger.info("Starting pipeline for %s on %s", model_name, device)
    ConsoleIO.print_header(f"Training: {model_name}")
    _announce_device(device)
    _log_memory_stats("Initial")

    dims = extract_tower_dims(cfg)

    df = load_and_clean_data(csv_path, cfg)
    logger.info("Loaded %d samples with %d columns", len(df), len(df.columns))
    ConsoleIO.print_info(f"Dataset: {len(df)} samples")

    train_df, val_df = stratified_split(df, validation_split)
    logger.info("Split: %d train, %d validation", len(train_df), len(val_df))
    ConsoleIO.print_info(f"Train: {len(train_df)} | Val: {len(val_df)}")

    train_dataset, val_dataset = build_two_tower_datasets(train_df, val_df, cfg, dims)
    drug_dim, geno_dim, target_dims = infer_dataset_dimensions(train_dataset, cfg)
    logger.info("Inferred dimensions: Drug=%d, Geno=%d", drug_dim, geno_dim)
    logger.info("Target dimensions: %s", target_dims)

    train_loader, val_loader = build_train_val_loaders(
        train_dataset, val_dataset, batch_size
    )

    model = build_gnn_model(
        model_name=model_name,
        dims=dims,
        drug_dim=drug_dim,
        geno_dim=geno_dim,
        target_dims=target_dims,
        params=cfg.params,
        device=device,
    )
    num_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %d", num_params)
    ConsoleIO.print_success(f"Model created: {num_params:,} parameters")

    _persist_training_artifacts(
        model_name=model_name,
        encoders=train_dataset.target_encoder.encoders,
        drug_dim=drug_dim,
        geno_dim=geno_dim,
    )

    trainer = _setup_trainer(model, cfg, device, model_name)
    _execute_training(trainer, train_loader, val_loader, epochs, patience)


# --------------------------------------------------------------------------- #
# Helpers local to the pipeline (everything else moved to engine.base).
# --------------------------------------------------------------------------- #


def _announce_device(device: torch.device) -> None:
    if device.type == "cuda":
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.debug("GPU detected: %s (%.1f GB)", name, mem)
        ConsoleIO.print_success(f"Using GPU: {name}")
    else:
        logger.warning("No GPU detected, using CPU (training will be slower)")
        ConsoleIO.print_warning("No GPU detected - using CPU")


def _log_memory_stats(stage: str) -> None:
    if not torch.cuda.is_available():
        return
    allocated_gb = torch.cuda.memory_allocated() / 1024**3
    reserved_gb = torch.cuda.memory_reserved() / 1024**3
    logger.info(
        "Memory (%s): allocated=%.2f GB reserved=%.2f GB",
        stage,
        allocated_gb,
        reserved_gb,
    )


def _persist_training_artifacts(
    *,
    model_name: str,
    encoders: dict,
    drug_dim: int,
    geno_dim: int,
) -> None:
    """Persist what the inference path needs to reconstruct the same model.

    Bundles the fitted target encoders together with the per-tower feature
    widths that were actually inferred from the training graphs. Predictor
    needs both: encoders to decode logits and the dims to build a model
    whose ``in_features`` match the saved ``state_dict``.
    """
    enc_dir = get_settings().paths.encoders
    enc_dir.mkdir(parents=True, exist_ok=True)
    enc_path = enc_dir / f"encoders_{model_name}.pkl"
    bundle = {
        "encoders": encoders,
        "drug_dim": int(drug_dim),
        "geno_dim": int(geno_dim),
        "schema_version": 1,
    }
    joblib.dump(bundle, enc_path)
    logger.info(
        "Persisted training artifacts (encoders=%d, drug_dim=%d, geno_dim=%d) to %s",
        len(encoders),
        drug_dim,
        geno_dim,
        enc_path,
    )


def _setup_trainer(
    model: torch.nn.Module,
    cfg,
    device: torch.device,
    model_name: str,
) -> StandardTrainer:
    uncertainty_net = LossFactory.create_uncertainty_wrapper(
        tasks=cfg.targets, device=device
    )
    optimizer = OptimizerFactory.create(
        model=model, params=cfg.params, uncertainty_module=uncertainty_net
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=8, factor=0.5
    )
    trainer = StandardTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        target_cols=cfg.targets,
        multi_label_cols=get_settings().multi_label_set,
        params=cfg.params,
        uncertainty_module=uncertainty_net,
        checkpoint_name=model_name,
    )
    logger.info("Trainer initialized")
    return trainer


def _execute_training(
    trainer: StandardTrainer,
    train_loader,
    val_loader,
    epochs: int,
    patience: int,
) -> None:
    logger.info("Starting training for %d epochs (patience=%d)", epochs, patience)
    ConsoleIO.print_header(f"Training: {epochs} epochs")
    try:
        trainer.fit(train_loader, val_loader, epochs=epochs, patience=patience)
        logger.info("Training completed successfully")
        ConsoleIO.print_success("Training completed!")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error("Out of memory during training")
            ConsoleIO.print_error("Out of memory - try reducing batch size")
        raise
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    train_pipeline(
        model_name="TwoTowerGAT",
        csv_path="data/processed/training_data.tsv",
        epochs=50,
        batch_size=128,
    )
