# Pharmagen - Training Pipeline
# Orchestrates data loading, model setup, and training loop.

import logging
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

from src.config.manager import get_model_config, DIRS, SEED, MULTI_LABEL_COLS
from src.data.loaders import DataLoaderUtils, DoubleTowerCollater, DoubleTowerDataset
from src.modeling.architectures.layers import create_gnn_model
from src.modeling.engine.trainer import PGenTrainer
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

def train_pipeline(model_name: str, csv_path: str, epochs: int = 50, batch_size: int = 32):
    """
    Standard training pipeline (Non-Optuna).
    """
    # 1. Configuration & Setup
    cfg = get_model_config(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting pipeline for {model_name} on {device}")
    
    # 2. Data Loading (Raw)
    df = DataLoaderUtils.load_dataframe(csv_path)
    
    # Stratified Split (if applicable)
    stratify = df["_stratify"] if "_stratify" in df.columns else None
    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=stratify, random_state=SEED
    )
    
    # 3. Dataset Construction
    # We must fit encoders on Train and reuse on Val
    logger.info("Initializing Train Dataset...")
    train_dataset = DoubleTowerDataset(
        df=train_df,
        drug_col=cfg.get("drug_col", "compound_id"),
        haplo_col=cfg.get("haplo_col", "genotype_id"),
        target_cols=cfg["targets"],
        multilabel_cols=list(MULTI_LABEL_COLS),
        preload_ram=False # Set True if you have >32GB RAM
    )
    
    logger.info("Initializing Validation Dataset...")
    # PASS TRAIN ENCODERS TO VAL DATASET
    val_dataset = DoubleTowerDataset(
        df=val_df,
        drug_col=cfg.get("drug_col", "compound_id"),
        haplo_col=cfg.get("haplo_col", "genotype_id"),
        target_cols=cfg["targets"],
        multilabel_cols=list(MULTI_LABEL_COLS),
        encoders=train_dataset.encoders, # <--- CRITICAL: Reuse encoders
        preload_ram=False
    )
    
    # 4. Dimension Calculation (Peek at data)
    sample_data = train_dataset[0]
    drug_dim = sample_data['drug_data'].x.shape[1]
    haplo_dim = sample_data['haplo_data'].x.shape[1]
    
    target_dims = {}
    for col in cfg["targets"]:
        # For MultiLabelBinarizer/LabelEncoder, classes_ holds the labels
        target_dims[col] = len(train_dataset.encoders[col].classes_)
        
    logger.info(f"Dimensions Detected -> Drug: {drug_dim}, Haplo: {haplo_dim}")
    logger.info(f"Target Dimensions: {target_dims}")

    # 5. DataLoaders
    collater = DoubleTowerCollater()
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        collate_fn=collater, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        collate_fn=collater, num_workers=4, pin_memory=True
    )

    # 6. Model Initialization
    drug_config = {'num_features': drug_dim, 'edge_dim': 0} # Update edge_dim if using edge attrs
    haplo_config = {'num_features': haplo_dim, 'edge_dim': 0}

    model = create_gnn_model(
        model_name=model_name,
        drug_config=drug_config,
        haplo_config=haplo_config,
        target_dims=target_dims,
        params=cfg["params"]
    ).to(device)

    # 7. Trainer Setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["params"]["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
    
    trainer = PGenTrainer(
        model=model, 
        optimizer=optimizer, 
        scheduler=scheduler,
        device=device,
        target_cols=cfg["targets"],
        multi_label_cols=MULTI_LABEL_COLS,
        params=cfg["params"]
    )
    
    # 8. Execute Training
    trainer.fit(train_loader, val_loader, epochs=epochs, patience=10)

if __name__ == "__main__":
    # Example usage
    # Ensure logging is configured
    logging.basicConfig(level=logging.INFO)
    train_pipeline("TwoTowerGAT", "data/processed/training_data.tsv", epochs=50)