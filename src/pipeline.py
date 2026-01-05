# Pharmagen - Training Pipeline
# Orchestrates data loading, model setup, and training loop.

import logging
from pathlib import Path
from typing import Union

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.config.manager import MULTI_LABEL_COLS, SEED, get_model_config
from src.data.collator import DoubleTowerCollater
from src.data.datasets import DoubleTowerDataset
from src.modeling.architectures.layers import create_gnn_model
from src.modeling.engine.trainer import PGenTrainer
from src.utils.io import DataLoaderUtils
from src.utils.memory import MemoryMonitor, estimate_batch_memory_mb, estimate_model_memory_mb
from src.utils.module_builder import LossFactory, OptimizerFactory

logger = logging.getLogger(__name__)


def train_pipeline(
    model_name: str, csv_path: str | Path, epochs: int = 50, batch_size: int = 32
):
    """
    Standard training pipeline with memory monitoring and validation.
    
    Args:
        model_name: Name of the model to train.
        csv_path: Path to training data CSV/TSV.
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        
    Raises:
        ValueError: If configuration is invalid.
        RuntimeError: If insufficient memory is available.
    """
    # 1. Configuration & Setup
    cfg = get_model_config(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting pipeline for {model_name} on {device}")
    
    # Log initial memory state
    MemoryMonitor.log_memory_stats("Initial - ")
    
    dims = {}
    dims["drug_feat"] = cfg.get("drug_in_features", 25)
    dims["drug_edge"] = cfg.get("drug_edge_dim", 7)
    dims["haplo_feat"] = cfg.get("haplo_in_features", 9)
    dims["haplo_edge"] = cfg.get("haplo_edge_dim", 3)

    # 2. Data Loading (Raw)
    df = DataLoaderUtils.load_dataframe(
        csv_path, cols=cfg["cols"], stratify_col=cfg.get("stratify_col", None)
    )

    logger.debug(
        f"Loaded dataframe with cols: {df.columns.tolist()} and shape: {df.shape}"
    )
    
    # Validate dataset size
    if len(df) < 100:
        raise ValueError(f"Dataset too small: {len(df)} samples. Need at least 100.")
    
    # Estimate memory requirements
    estimated_batch_mem = estimate_batch_memory_mb(
        batch_size=batch_size,
        avg_nodes_per_graph=50,
        node_features=dims["drug_feat"],
        num_graphs=2
    )
    logger.info(f"Estimated batch memory: {estimated_batch_mem:.1f}MB")

    # Stratified Split (if applicable)
    stratify = df["_stratify"] if "_stratify" in df.columns else None
    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=stratify, random_state=SEED
    )
    
    logger.info(f"Split: {len(train_df)} train, {len(val_df)} validation samples")

    # 3. Dataset Construction
    # Determine if we should preload based on dataset size
    should_preload = len(train_df) < 5000  # Only preload if dataset is small
    if should_preload:
        logger.info("Small dataset detected, enabling RAM preloading for speed")
    else:
        logger.info("Large dataset detected, using disk-based loading to save memory")
    
    logger.info("Initializing Train Dataset...")
    train_dataset = DoubleTowerDataset(
        df=train_df,
        drug_col=cfg["features"][0],
        haplo_col=cfg["features"][1],
        target_cols=cfg["targets"],
        multilabel_cols=list(MULTI_LABEL_COLS),
        preload_ram=should_preload,
        input_dimensions=dims,
    )

    logger.info("Initializing Validation Dataset...")
    val_dataset = DoubleTowerDataset(
        df=val_df,
        drug_col=cfg["features"][0],
        haplo_col=cfg["features"][1],
        target_cols=cfg["targets"],
        multilabel_cols=list(MULTI_LABEL_COLS),
        encoders=train_dataset.encoders,  # <--- CRITICAL: Reuse encoders
        preload_ram=should_preload,
    )
    
    MemoryMonitor.log_memory_stats("After dataset creation - ")

    # 4. Dimension Calculation (Peek at data)
    sample_data = train_dataset[0]
    drug_dim = sample_data["drug_data"].x.shape[1]  # type: ignore
    haplo_dim = sample_data["haplo_data"].x.shape[1]  # type: ignore

    target_dims = {}
    for col in cfg["targets"]:
        target_dims[col] = len(train_dataset.encoders[col].classes_)

    logger.info(f"Dimensions Detected -> Drug: {drug_dim}, Haplo: {haplo_dim}")
    logger.info(f"Target Dimensions: {target_dims}")

    # 5. DataLoaders
    # Adjust num_workers based on available resources
    num_workers = 4 if not should_preload else 2  # Fewer workers if data is in RAM
    
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

    # 6. Model Initialization
    drug_edge_dim = cfg.get("drug_edge_dim", 0)
    haplo_edge_dim = cfg.get("haplo_edge_dim", 0)

    drug_config = {"num_features": drug_dim, "edge_dim": drug_edge_dim}
    haplo_config = {"num_features": haplo_dim, "edge_dim": haplo_edge_dim}

    logger.debug(f"Drug Config: {drug_config}")
    logger.debug(f"Haplo Config: {haplo_config}")
    logger.debug(f"Model Params: {cfg.get('params', {})}")

    model = create_gnn_model(
        model_name=model_name,
        drug_config=drug_config,
        haplo_config=haplo_config,
        target_dims=target_dims,
        params=cfg["params"],
    ).to(device)
    
    # Log model size
    num_params = sum(p.numel() for p in model.parameters())
    model_mem = estimate_model_memory_mb(num_params)
    logger.info(f"Model parameters: {num_params:,}, estimated memory: {model_mem:.1f}MB")
    
    MemoryMonitor.log_memory_stats("After model creation - ")

    # 7. Trainer Setup
    uncertainty_net = LossFactory.create_uncertainty_wrapper(
        tasks=cfg["targets"], device=device
    )

    optimizer = OptimizerFactory.create(
        model=model, params=cfg["params"], uncertainty_module=uncertainty_net
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=8
    )

    trainer = PGenTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        target_cols=cfg["targets"],
        multi_label_cols=MULTI_LABEL_COLS,
        params=cfg["params"],
        uncertainty_module=uncertainty_net if uncertainty_net else None,
    )

    # 8. Execute Training
    logger.info(f"Starting training for {epochs} epochs...")
    try:
        trainer.fit(train_loader, val_loader, epochs=epochs, patience=10)
        logger.info("Training completed successfully")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error(
                "Out of memory error during training. "
                "Try reducing batch_size or using gradient accumulation."
            )
            MemoryMonitor.log_memory_stats("OOM Error - ")
        raise
    finally:
        # Cleanup
        MemoryMonitor.clear_memory(device=device, aggressive=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_pipeline("TwoTowerGAT", "data/processed/training_data.tsv", epochs=50)
