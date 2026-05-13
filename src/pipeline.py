# src/pipeline.py
# Pharmagen - Training Pipeline
# Orchestrates data loading, model setup, and training loop.
import gc
import logging
import os
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.config import get_model_config, get_settings
from src.data.cleaning import PharmacogenomicCleaner
from src.data.collator import DoubleTowerCollater
from src.data.datasets import DoubleTowerDataset
from src.data.loaders import TabularLoader
from src.core import (
    ConfigurationError,
    ConfigValidator,
    DataError,
    DataValidator,
    ModelError,
)
from src.interface.ui import ConsoleIO
from src.model.architectures.layers import create_gnn_model
from src.model.factories import LossFactory, OptimizerFactory
from src.model.training.standard import StandardTrainer

logger = logging.getLogger(__name__)

# Constants
MIN_DATASET_SIZE = 100
PRELOAD_THRESHOLD = 10000
DEFAULT_NUM_WORKERS = 4
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

    Args:
        model_name: Name of the model configuration in models.toml.
        csv_path: Path to training data (CSV/TSV format).
        epochs: Number of training epochs.
        batch_size: Batch size for DataLoader.
        validation_split: Fraction of data for validation (0.1-0.3).
        patience: Early stopping patience.

    Raises:
        ConfigurationError: If model configuration is invalid.
        DataError: If data is invalid or incompatible.
        PharmagenMemoryError: If insufficient memory is available.
        ModelError: If model creation fails.
    """

    # 1. Configuration Loading, Validation and Device Setup

    try:
        cfg = get_model_config(model_name)
    except Exception as e:
        raise ConfigurationError(
            f"Failed to load configuration for '{model_name}': {e}"
        ) from e

    if not MIN_VALIDATION_SPLIT <= validation_split <= MAX_VALIDATION_SPLIT:
        raise ConfigurationError(
            f"validation_split must be in [{MIN_VALIDATION_SPLIT}, {MAX_VALIDATION_SPLIT}], "
            f"got {validation_split}"
        )

    device = _setup_device()
    logger.info("Starting pipeline for %s on %s", model_name, device)
    ConsoleIO.print_header(f"Training:  {model_name}")

    MemoryMonitor.log_memory_stats("Initial")

    dims = _extract_dimensions(cfg)

    # 2. Data Loading and Validation
    df = _load_and_validate_data(csv_path, cfg)

    # 3. Data Splitting
    train_df, val_df = _split_data(df, validation_split, cfg)

    # 4. Dataset Creation
    train_dataset, val_dataset = _create_datasets(train_df, val_df, cfg, dims)

    # 5. Dimension inference
    drug_dim, geno_dim, target_dims = _infer_dimensions(train_dataset, cfg)

    # 6. DataLoaders
    train_loader, val_loader = _create_dataloaders(train_dataset, val_dataset, batch_size)

    # 7. Model Initialization
    model = _create_and_validate_model(
        model_name, cfg, dims, drug_dim, geno_dim, target_dims, device
    )

    # 8. Trainer initialization
    trainer = _setup_trainer(model, cfg, device, model_name)

    # 9. Training Execution
    _execute_training(trainer, train_loader, val_loader, epochs, patience, device)

# =============================================================================
# HELPER FUNCTIONS (Private API)
# =============================================================================

def _setup_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.debug("GPU detected: %s (%.1f GB)", gpu_name, gpu_mem)
        ConsoleIO.print_success(f"Using GPU: {gpu_name}")
    else:
        device = torch.device("cpu")
        logger.warning("No GPU detected, using CPU (training will be slower)")
        ConsoleIO.print_warning("No GPU detected - using CPU")

    return device


def _extract_dimensions(cfg) -> dict:
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

    for graph_type, dimensions in dims.items():
        for dim_name, dim_value in dimensions.items():
            if not isinstance(dim_value, int) or dim_value < 0:
                raise ConfigurationError(
                    f"Invalid dimension '{graph_type}.{dim_name}': {dim_value} "
                    "(must be >= 0 integer)"
                )

    logger.debug("Dimensions: %s", dims)
    return dims


def _load_and_validate_data(csv_path: str | Path, cfg):
    try:
        raw_df = TabularLoader.load(csv_path, columns=cfg.cols or None)
        df = PharmacogenomicCleaner(
            multi_label_cols=get_settings().multi_label_set,
        ).clean(raw_df, stratify_col=cfg.stratify_col)
    except Exception as e:
        raise DataError(f"Failed to load data from {csv_path}: {e}") from e

    logger.info("Loaded %d samples with %d columns", len(df), len(df.columns))
    ConsoleIO.print_info(f"Dataset:  {len(df)} samples")

    if len(df) < MIN_DATASET_SIZE:
        raise DataError(
            f"Dataset too small: {len(df)} samples (minimum: {MIN_DATASET_SIZE})"
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
        ConsoleIO.print_warning("Some columns have >10% missing values")

    for target in cfg.targets:
        if target in df.columns:
            class_counts = DataValidator.check_class_balance(
                df, target_column=target, min_samples_per_class=10
            )
            logger.info("Target %r distribution: %s", target, class_counts)

    return df


def _split_data(df, validation_split: float, cfg):
    seed = get_settings().seed
    stratify = df["_stratify"] if "_stratify" in df.columns else None

    train_df, val_df = train_test_split(
        df, test_size=validation_split, stratify=stratify, random_state=seed
    )

    logger.info("Split: %d train, %d validation", len(train_df), len(val_df))
    ConsoleIO.print_info(f"Train: {len(train_df)} | Val: {len(val_df)}")

    return train_df, val_df


def _create_datasets(train_df, val_df, cfg, dims):
    multi_label_cols = get_settings().multi_label_set
    should_preload = len(train_df) < PRELOAD_THRESHOLD

    if should_preload:
        logger.info("Small dataset: enabling RAM preloading")
        ConsoleIO.print_info("Using RAM preloading (fast)")
    else:
        logger.info("Large dataset: using disk-based loading")
        ConsoleIO.print_info("Using disk loading (memory-efficient)")

    logger.info("Initializing training dataset...")
    train_dataset = DoubleTowerDataset(
        df=train_df,
        drug_col=cfg.features[0],
        geno_col=cfg.features[1],
        target_cols=cfg.targets,
        multilabel_cols=list(multi_label_cols),
        preload_ram=should_preload,
        input_dimensions=dims,
    )

    logger.info("Initializing validation dataset...")
    val_dataset = DoubleTowerDataset(
        df=val_df,
        drug_col=cfg.features[0],
        geno_col=cfg.features[1],
        target_cols=cfg.targets,
        multilabel_cols=list(multi_label_cols),
        encoders=train_dataset.target_encoder.encoders,
        preload_ram=should_preload,
    )

    return train_dataset, val_dataset


def _infer_dimensions(train_dataset, cfg):
    try:
        sample_data = train_dataset[0]
        drug_dim = sample_data["drug_data"].x.shape[1]
        geno_dim = sample_data["geno_data"].x.shape[1]
    except Exception as e:
        raise DataError(f"Failed to infer dimensions from dataset: {e}") from e

    if drug_dim <= 0 or geno_dim <= 0:
        raise DataError(f"Invalid inferred dimensions: drug={drug_dim}, geno={geno_dim}")

    encoders = train_dataset.target_encoder.encoders
    target_dims = {}
    for col in cfg.targets:
        if col in encoders:
            target_dims[col] = len(encoders[col].classes_)
        else:
            raise DataError(f"Target column '{col}' not encoded")

    logger.info("Inferred dimensions: Drug=%d, Geno=%d", drug_dim, geno_dim)
    logger.info("Target dimensions: %s", target_dims)

    return drug_dim, geno_dim, target_dims


def _create_dataloaders(train_dataset, val_dataset, batch_size: int):
    should_preload = len(train_dataset) < PRELOAD_THRESHOLD
    cpu_count = os.cpu_count() or 1
    num_workers = min(DEFAULT_NUM_WORKERS, cpu_count - 1) if not should_preload else 2

    logger.info("DataLoader workers: %d", num_workers)

    collater = DoubleTowerCollater()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collater,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collater,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader


def _create_and_validate_model(model_name, cfg, dims, drug_dim, geno_dim, target_dims, device):
    drug_config = {"num_features": drug_dim, "edge_dim": dims["drugs"]["edges"]}
    geno_config = {"num_features": geno_dim, "edge_dim": dims["geno"]["edges"]}

    logger.debug("Drug config: %s", drug_config)
    logger.debug("Geno config: %s", geno_config)

    try:
        model = create_gnn_model(
            model_name=model_name,
            drug_config=drug_config,
            geno_config=geno_config,
            target_dims=target_dims,
            params=cfg.params,
        ).to(device)
    except Exception as e:
        raise ModelError(f"Failed to create model '{model_name}': {e}") from e

    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: %d (trainable: %d)", num_params, trainable_params)
    ConsoleIO.print_success(f"Model created: {num_params:,} parameters")
    return model


def _setup_trainer(model, cfg, device, model_name: str = "training_session"):
    uncertainty_net = LossFactory.create_uncertainty_wrapper(
        tasks=cfg.targets, device=device
    )

    optimizer = OptimizerFactory.create(
        model=model, params=cfg.params, uncertainty_module=uncertainty_net
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=8, factor=0.5,
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


def _execute_training(trainer, train_loader, val_loader, epochs, patience, device):
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


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

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
