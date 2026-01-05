#!/usr/bin/env python3
# Pharmagen - Performance Benchmark Script
# Copyright (C) 2025 Adrim Hamed Outmani
#
# This script benchmarks training performance to measure optimization impact.

import argparse
import logging
import time
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split

from src.config.manager import SEED, get_model_config
from src.data.loaders import DoubleTowerCollater, DoubleTowerDataset
from src.utils.io import DataLoaderUtils
from src.utils.performance import (
    apply_performance_optimizations,
    log_gpu_info,
    log_training_config,
    PerformanceMonitor,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def benchmark_data_loading(
    model_name: str,
    csv_path: Path,
    batch_size: int = 32,
    num_workers: int = 8,
    preload_ram: bool = True
):
    """
    Benchmark data loading performance.
    
    Args:
        model_name: Name of the model configuration
        csv_path: Path to training data
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes
        preload_ram: Whether to preload data into RAM
    """
    logger.info("=" * 70)
    logger.info("Data Loading Benchmark")
    logger.info("=" * 70)
    logger.info(f"Model: {model_name}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Num workers: {num_workers}")
    logger.info(f"Preload RAM: {preload_ram}")
    logger.info("=" * 70)
    
    # Load configuration
    cfg = get_model_config(model_name)
    
    # Load data
    df = DataLoaderUtils.load_dataframe(
        csv_path, cols=cfg["cols"], stratify_col=cfg.get("stratify_col", None)
    )
    
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=SEED)
    logger.info(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")
    
    # Create dataset
    dims = {
        "drug_feat": cfg.get("drug_in_features", 25),
        "drug_edge": cfg.get("drug_edge_dim", 7),
        "haplo_feat": cfg.get("haplo_in_features", 9),
        "haplo_edge": cfg.get("haplo_edge_dim", 3),
    }
    
    monitor = PerformanceMonitor()
    
    with monitor.measure("dataset_creation"):
        train_dataset = DoubleTowerDataset(
            df=train_df,
            drug_col=cfg["features"][0],
            haplo_col=cfg["features"][1],
            target_cols=cfg["targets"],
            multilabel_cols=list(cfg.get("multi_label_cols", [])),
            preload_ram=preload_ram,
            input_dimensions=dims,
        )
    
    # Create dataloader
    from torch.utils.data import DataLoader
    collater = DoubleTowerCollater()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collater,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=3 if num_workers > 0 else None,
    )
    
    # Benchmark iteration
    logger.info("\nBenchmarking data loading (5 batches)...")
    num_batches = 5
    
    with monitor.measure("data_loading"):
        for i, batch in enumerate(train_loader):
            if i >= num_batches:
                break
            # Just access the data to ensure it's loaded
            _ = batch["drug_batch"]
            _ = batch["haplo_batch"]
            _ = batch["targets"]
    
    # Print results
    logger.info("\n" + monitor.report())
    
    return monitor


def benchmark_model_forward(
    model_name: str,
    csv_path: Path,
    batch_size: int = 32,
    device: str = "cuda"
):
    """
    Benchmark model forward pass performance.
    
    Args:
        model_name: Name of the model configuration
        csv_path: Path to training data
        batch_size: Batch size for testing
        device: Device to run on ('cuda' or 'cpu')
    """
    logger.info("=" * 70)
    logger.info("Model Forward Pass Benchmark")
    logger.info("=" * 70)
    logger.info(f"Model: {model_name}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Device: {device}")
    logger.info("=" * 70)
    
    # Apply optimizations
    apply_performance_optimizations()
    
    # Load configuration
    cfg = get_model_config(model_name)
    device_obj = torch.device(device)
    
    # Create a sample dataset
    df = DataLoaderUtils.load_dataframe(
        csv_path, cols=cfg["cols"], stratify_col=cfg.get("stratify_col", None)
    )
    train_df, _ = train_test_split(df, test_size=0.2, random_state=SEED)
    
    dims = {
        "drug_feat": cfg.get("drug_in_features", 25),
        "drug_edge": cfg.get("drug_edge_dim", 7),
        "haplo_feat": cfg.get("haplo_in_features", 9),
        "haplo_edge": cfg.get("haplo_edge_dim", 3),
    }
    
    train_dataset = DoubleTowerDataset(
        df=train_df,
        drug_col=cfg["features"][0],
        haplo_col=cfg["features"][1],
        target_cols=cfg["targets"],
        multilabel_cols=list(cfg.get("multi_label_cols", [])),
        preload_ram=True,
        input_dimensions=dims,
    )
    
    # Get dimensions
    sample_data = train_dataset[0]
    drug_dim = sample_data["drug_data"].x.shape[1]
    haplo_dim = sample_data["haplo_data"].x.shape[1]
    
    target_dims = {}
    for col in cfg["targets"]:
        target_dims[col] = len(train_dataset.encoders[col].classes_)
    
    # Create model
    from src.modeling.architectures.layers import create_gnn_model
    
    model = create_gnn_model(
        model_name=model_name,
        drug_config={"num_features": drug_dim, "edge_dim": cfg.get("drug_edge_dim", 0)},
        haplo_config={"num_features": haplo_dim, "edge_dim": cfg.get("haplo_edge_dim", 0)},
        target_dims=target_dims,
        params=cfg["params"],
    ).to(device_obj)
    
    # Create dataloader
    from torch.utils.data import DataLoader
    collater = DoubleTowerCollater()
    
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collater,
        num_workers=0,  # Single process for accurate timing
        pin_memory=True,
    )
    
    # Warm-up
    logger.info("\nWarming up GPU...")
    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        drug_data = batch["drug_batch"].to(device_obj)
        haplo_data = batch["haplo_batch"].to(device_obj)
        _ = model(drug_data, haplo_data)
    
    # Benchmark
    monitor = PerformanceMonitor()
    num_batches = 10
    
    logger.info(f"\nBenchmarking forward pass ({num_batches} batches)...")
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
            
            with monitor.measure("forward_pass"):
                drug_data = batch["drug_batch"].to(device_obj)
                haplo_data = batch["haplo_batch"].to(device_obj)
                _ = model(drug_data, haplo_data)
    
    # Print results
    logger.info("\n" + monitor.report())
    
    return monitor


def main():
    parser = argparse.ArgumentParser(description="Pharmagen Performance Benchmark")
    parser.add_argument(
        "--model",
        type=str,
        default="TwoTowerGAT",
        help="Model name to benchmark"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("train_data/train_data.tsv"),
        help="Path to training data"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for benchmarking"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of DataLoader workers"
    )
    parser.add_argument(
        "--benchmark",
        choices=["data", "model", "all"],
        default="all",
        help="Which benchmark to run"
    )
    
    args = parser.parse_args()
    
    # Log system info
    logger.info("System Information:")
    log_gpu_info()
    log_training_config()
    
    # Run benchmarks
    if args.benchmark in ["data", "all"]:
        logger.info("\n\n")
        benchmark_data_loading(
            args.model,
            args.data,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            preload_ram=True
        )
    
    if args.benchmark in ["model", "all"]:
        logger.info("\n\n")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        benchmark_model_forward(
            args.model,
            args.data,
            batch_size=args.batch_size,
            device=device
        )
    
    logger.info("\n" + "=" * 70)
    logger.info("Benchmark Complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
