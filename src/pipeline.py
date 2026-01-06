# src/pipeline.py
# Pharmagen - Training Pipeline
# Orchestrates data loading, model setup, and training loop.
import logging
import os
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.config.manager import (
    MULTI_LABEL_COLS,
    SEED,
    get_model_config,
)
from src.data.collator import DoubleTowerCollater
from src.data.datasets import DoubleTowerDataset
from src.interface.ui import ConsoleIO
from src.modeling.architectures.layers import create_gnn_model
from src.modeling.engine.trainer import PGenTrainer
from src.utils.exceptions import (
    ConfigurationError,
    DataError,
    ModelError,
)
from src.utils.exceptions import (
    MemoryError as PharmagenMemoryError,
)
from src.utils.io import DataLoaderUtils
from src.utils.memory import (
    MemoryMonitor,
    estimate_batch_memory_mb,
    estimate_model_memory_mb,
)
from src.utils.module_builder import LossFactory, OptimizerFactory
from src.utils.validation import ConfigValidator, DataValidator

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
    """
    Standard training pipeline with comprehensive validation and monitoring.

    This function orchestrates the complete training workflow:
    1. Configuration loading and validation
    2. Data loading and quality checks
    3. Dataset creation with automatic optimization
    4. Model initialization with memory estimation
    5. Training execution with monitoring
    6. Cleanup and resource management

    Args:
        model_name:  Name of the model configuration in models.toml.
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

    Example:
        >>> train_pipeline("TwoTowerGAT", "data/train.tsv", epochs=100)
    """

    # 1. Configuration Loading, Validation and Device Setup

    try:
        cfg = get_model_config(model_name)
    except Exception as e:
        raise ConfigurationError(
            f"Failed to load configuration for '{model_name}': {e}"
        ) from e

    # Validate validation split
    if not MIN_VALIDATION_SPLIT <= validation_split <= MAX_VALIDATION_SPLIT:
        raise ConfigurationError(
            f"validation_split must be in [{MIN_VALIDATION_SPLIT}, {MAX_VALIDATION_SPLIT}], "
            f"got {validation_split}"
        )

    device = _setup_device()
    logger.info(f"🚀 Starting pipeline for {model_name} on {device}")
    ConsoleIO.print_header(f"Training:  {model_name}")

    # Log initial memory state
    MemoryMonitor.log_memory_stats("Initial - ")

    dims = _extract_dimensions(cfg)
    """
    dims = {}
    dims["drug_feat"] = cfg.get("drug_in_features", 25)
    dims["drug_edge"] = cfg.get("drug_edge_dim", 7)
    dims["haplo_feat"] = cfg.get("haplo_in_features", 9)
    dims["haplo_edge"] = cfg.get("haplo_edge_dim", 3)
    """

    # 2. Data Loading and Validation
    df = _load_and_validate_data(csv_path, cfg)

    # 3. Data Splitting
    train_df, val_df = _split_data(df, validation_split)

    # 4. MEMORY ESTIMATION
    _estimate_and_log_memory(batch_size, dims)

    # 5. Dataset Creation
    train_dataset, val_dataset = _create_datasets(
        train_df, val_df, cfg, dims, MULTI_LABEL_COLS
    )

    MemoryMonitor.log_memory_stats("After dataset creation - ")

    # 6. Dimension inference
    drug_dim, haplo_dim, target_dims = _infer_dimensions(train_dataset, cfg)

    # 7. DataLoaders
    train_loader, val_loader = _create_dataloaders(
        train_dataset, val_dataset, batch_size
    )

    # 8. Model Initialization
    model = _create_and_validate_model(
        model_name, cfg, dims, drug_dim, haplo_dim, target_dims, device
    )

    MemoryMonitor.log_memory_stats("After model creation")

    # 9. Trainer initialization
    trainer = _setup_trainer(model, cfg, device)

    # 10. Training Execution
    _execute_training(trainer, train_loader, val_loader, epochs, patience, device)

# =============================================================================
# HELPER FUNCTIONS (Private API)
# =============================================================================

def _setup_device() -> torch.device:
    """Setup and validate compute device.

    Returns:
        torch.device: Configured device (CUDA or CPU).

    Raises:
        ConfigurationError: If CUDA is requested but not available.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.debug(f"🎮 GPU detected: {gpu_name} ({gpu_mem:.1f} GB)")
        ConsoleIO.print_success(f"Using GPU: {gpu_name}")
    else:
        device = torch.device("cpu")
        logger.warning("⚠️ No GPU detected, using CPU (training will be slower)")
        ConsoleIO.print_warning("No GPU detected - using CPU")

    return device


def _extract_dimensions(cfg: dict) -> dict:
    """Extract dimension configuration.

    Args:
        cfg:  Model configuration dictionary.

    Returns:
        Dictionary with dimension keys.
    """
    dims = {
        "drugs": {
            "features": cfg.get("drug_in_features", 25),
            "edges": cfg.get("drug_edge_dim", 7),
            "attrs": cfg.get("drug_in_attributes", 0)
        },
        "geno": {"features": cfg.get("haplo_in_features", 9),
                 "edges": cfg.get("haplo_edge_dim", 3),
                 "attrs": cfg.get("haplo_in_attributes", 0),
            },
        }

    # Validate dimensions
    for graph_type, dimensions in dims.items():
        for dim_name, dim_value in dimensions.items():
            if not isinstance(dim_value, int) or dim_value < 0:
                raise ConfigurationError(
                    f"Invalid dimension '{graph_type}.{dim_name}': {dim_value} "
                    "(must be >= 0 integer)"
                )

    logger.debug(f"Dimensions:  {dims}")
    return dims


def _load_and_validate_data(csv_path: str | Path, cfg: dict):
    """Load data and perform comprehensive validation.

    Args:
        csv_path: Path to data file.
        cfg: Model configuration.

    Returns:
        Validated DataFrame.

    Raises:
        DataError: If data is invalid.
    """
    try:
        df = DataLoaderUtils.load_dataframe(
            csv_path, cols=cfg["cols"], stratify_col=cfg.get("stratify_col")
        )
    except Exception as e:
        raise DataError(f"Failed to load data from {csv_path}: {e}") from e

    logger.info(f"📊 Loaded {len(df)} samples with {len(df.columns)} columns")
    ConsoleIO.print_info(f"Dataset:  {len(df)} samples")

    # Validate dataset size
    if len(df) < MIN_DATASET_SIZE:
        raise DataError(
            f"Dataset too small: {len(df)} samples (minimum: {MIN_DATASET_SIZE})"
        )

    # Validate required columns
    try:
        ConfigValidator.validate_data_columns(
            df.columns.tolist(), cfg["features"], cfg["targets"]
        )
    except ValueError as e:
        raise DataError(str(e)) from e

    # Check data quality - missing values
    missing_stats = DataValidator.check_missing_values(
        df, cfg["features"] + cfg["targets"], threshold=0.5
    )

    if any(frac > 0.1 for frac in missing_stats.values()): # noqa
        logger.warning(f"High missing values detected: {missing_stats}")
        ConsoleIO.print_warning("Some columns have >10% missing values")

    # Check class balance for targets
    for target in cfg["targets"]:
        if target in df.columns:
            class_counts = DataValidator.check_class_balance(
                df, target_column=target, min_samples_per_class=10
            )
            logger.info(f"Target '{target}' distribution: {class_counts}")

    return df


def _split_data(df, validation_split: float):
    """Split data into train and validation sets.

    Args:
        df: Input DataFrame.
        validation_split: Fraction for validation.

    Returns:
        Tuple of (train_df, val_df).
    """
    stratify = df["_stratify"] if "_stratify" in df.columns else None

    train_df, val_df = train_test_split(
        df, test_size=validation_split, stratify=stratify, random_state=SEED
    )

    logger.info(f"📂 Split:  {len(train_df)} train, {len(val_df)} validation")
    ConsoleIO.print_info(f"Train: {len(train_df)} | Val: {len(val_df)}")

    return train_df, val_df


def _estimate_and_log_memory(batch_size:  int, dims: dict):
    """Estimate and log memory requirements.

    Args:
        batch_size: Batch size.
        dims: Dimension dictionary.
    """
    estimated_batch_mem = estimate_batch_memory_mb(
        batch_size=batch_size,
        avg_nodes_per_graph=50,
        node_features=dims["drugs"]["features"],
        num_graphs=2,
    )

    logger.info(f"💾 Estimated batch memory: {estimated_batch_mem:.1f} MB")

    # Warn if memory might be tight
    if torch.cuda.is_available():
        available_mem = (
            torch.cuda.get_device_properties(0).total_memory / 1024**2
        )  # MB
        if estimated_batch_mem > available_mem * 0.5:
            ConsoleIO.print_warning(
                f"Batch size may be too large ({estimated_batch_mem:.0f}MB estimated)"
            )


def _create_datasets(train_df, val_df, cfg, dims, multi_label_cols):
    """Create train and validation datasets.

    Args:
        train_df: Training DataFrame.
        val_df: Validation DataFrame.
        cfg: Model configuration.
        dims: Dimension dictionary.
        multi_label_cols: Multi-label column set.

    Returns:
        Tuple of (train_dataset, val_dataset).
    """
    # Determine RAM preloading strategy
    should_preload = len(train_df) < PRELOAD_THRESHOLD

    if should_preload:
        logger.info("📥 Small dataset:  enabling RAM preloading")
        ConsoleIO.print_info("Using RAM preloading (fast)")
    else:
        logger.info("💿 Large dataset: using disk-based loading")
        ConsoleIO.print_info("Using disk loading (memory-efficient)")

    logger.info("Initializing training dataset...")
    train_dataset = DoubleTowerDataset(
        df=train_df,
        drug_col=cfg["features"][0],
        haplo_col=cfg["features"][1],
        target_cols=cfg["targets"],
        multilabel_cols=list(multi_label_cols),
        preload_ram=should_preload,
        input_dimensions=dims,
    )

    logger.info("Initializing validation dataset...")
    val_dataset = DoubleTowerDataset(
        df=val_df,
        drug_col=cfg["features"][0],
        haplo_col=cfg["features"][1],
        target_cols=cfg["targets"],
        multilabel_cols=list(multi_label_cols),
        encoders=train_dataset.encoders,  # CRITICAL: Reuse encoders
        preload_ram=should_preload,
    )

    return train_dataset, val_dataset


def _infer_dimensions(train_dataset, cfg):
    """Infer actual dimensions from dataset.

    Args:
        train_dataset: Training dataset.
        cfg: Model configuration.

    Returns:
        Tuple of (drug_dim, haplo_dim, target_dims).

    Raises:
        DataError: If dimension inference fails.
    """
    try:
        sample_data = train_dataset[0]
        drug_dim = sample_data["drug_data"].x.shape[1]
        haplo_dim = sample_data["haplo_data"].x.shape[1]
    except Exception as e:
        raise DataError(f"Failed to infer dimensions from dataset: {e}") from e

    # Validate dimensions match expectations
    if drug_dim <= 0 or haplo_dim <= 0:
        raise DataError(f"Invalid inferred dimensions: drug={drug_dim}, haplo={haplo_dim}")

    target_dims = {}
    for col in cfg["targets"]:
        if col in train_dataset.encoders:
            target_dims[col] = len(train_dataset.encoders[col]. classes_)
        else:
            raise DataError(f"Target column '{col}' not encoded")

    logger.info(f"🔍 Inferred dimensions:  Drug={drug_dim}, Haplo={haplo_dim}")
    logger.info(f"🎯 Target dimensions: {target_dims}")

    return drug_dim, haplo_dim, target_dims


def _create_dataloaders(train_dataset, val_dataset, batch_size:  int):
    """Create optimized DataLoaders.

    Args:
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        batch_size: Batch size.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    # Intelligent num_workers selection
    should_preload = len(train_dataset) < PRELOAD_THRESHOLD
    cpu_count = os.cpu_count() or 1
    num_workers = min(DEFAULT_NUM_WORKERS, cpu_count - 1) if not should_preload else 2

    logger.info(f"⚙️ DataLoader workers: {num_workers}")

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


def _create_and_validate_model(
    model_name, cfg, dims, drug_dim, haplo_dim, target_dims, device
):
    """Create model and validate memory requirements.

    Args:
        model_name: Model name.
        cfg: Configuration.
        dims: Dimension dict.
        drug_dim: Drug feature dimension.
        haplo_dim: Haplotype feature dimension.
        target_dims: Target dimensions.
        device: Compute device.

    Returns:
        Initialized model on device.

    Raises:
        ModelError: If model creation fails.
        PharmagenMemoryError: If insufficient memory.
    """
    drug_config = {"num_features": drug_dim, "edge_dim": dims["drugs"]["edges"]}
    haplo_config = {"num_features": haplo_dim, "edge_dim": dims["geno"]["edges"]}

    logger. debug(f"Drug config: {drug_config}")
    logger.debug(f"Haplo config: {haplo_config}")

    try:
        model = create_gnn_model(
            model_name=model_name,
            drug_config=drug_config,
            haplo_config=haplo_config,
            target_dims=target_dims,
            params=cfg["params"],
        ).to(device)
    except Exception as e:
        raise ModelError(f"Failed to create model '{model_name}': {e}") from e

    # Log model statistics
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p. numel() for p in model.parameters() if p.requires_grad)
    model_mem = estimate_model_memory_mb(num_params)

    logger.info(f"🧠 Model parameters: {num_params:,} (trainable: {trainable_params:,})")
    logger.info(f"💾 Estimated model memory: {model_mem:.1f} MB")
    ConsoleIO.print_success(f"Model created: {num_params:,} parameters")

    # Check if model fits in memory
    if torch.cuda.is_available():
        available_mem = torch.cuda.get_device_properties(0).total_memory / 1024**2
        if model_mem > available_mem * 0.8:
            raise PharmagenMemoryError(
                f"Model too large for GPU:  {model_mem:.0f}MB required, "
                f"{available_mem:.0f}MB available"
            )

    return model


def _setup_trainer(model, cfg, device):
    """Setup trainer with optimizer and scheduler.

    Args:
        model: PyTorch model.
        cfg: Configuration.
        device: Compute device.

    Returns:
        Configured PGenTrainer.
    """
    uncertainty_net = LossFactory.create_uncertainty_wrapper(
        tasks=cfg["targets"], device=device
    )

    optimizer = OptimizerFactory.create(
        model=model, params=cfg["params"], uncertainty_module=uncertainty_net
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=8, factor=0.5,
    )

    trainer = PGenTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        target_cols=cfg["targets"],
        multi_label_cols=MULTI_LABEL_COLS,
        params=cfg["params"],
        uncertainty_module=uncertainty_net,
    )

    logger.info("✅ Trainer initialized")
    return trainer


def _execute_training(trainer, train_loader, val_loader, epochs, patience, device):
    """Execute training with error handling and cleanup.

    Args:
        trainer: PGenTrainer instance.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        epochs: Number of epochs.
        patience: Early stopping patience.
        device: Compute device.

    Raises:
        RuntimeError: If training fails.
    """
    logger.info(f"🏋️ Starting training for {epochs} epochs (patience={patience})...")
    ConsoleIO.print_header(f"Training: {epochs} epochs")

    try:
        trainer.fit(train_loader, val_loader, epochs=epochs, patience=patience)
        logger.info("✅ Training completed successfully")
        ConsoleIO.print_success("Training completed!")

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error("❌ Out of memory during training")
            ConsoleIO.print_error("Out of memory - try reducing batch size")
            MemoryMonitor.log_memory_stats("OOM Error")
        raise

    finally:
        # Always cleanup, even on error
        logger.info("🧹 Cleaning up resources...")
        MemoryMonitor.clear_memory(device=device, aggressive=True)
        logger.info("✅ Cleanup complete")


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
