import logging

import pandas as pd
import torch
from torch.utils.data import DataLoader

# Project Imports
from src.config.manager import MULTI_LABEL_COLS, get_model_config
from src.data.collator import DoubleTowerCollater
from src.data.datasets import DoubleTowerDataset
from src.modeling.architectures.layers import create_gnn_model

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("SanityCheck")


def run_sanity_check(model_name: str, csv_path: str, batch_size: int = 4):
    logger.info(f"--- Starting Sanity Check for {model_name} ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # 1. Load Data
    logger.info(f"1. Loading Data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path, sep="\t" if csv_path.endswith(".tsv") else ",")
        # Take a tiny subsample to speed things up
        df_sample = df.head(10).reset_index(drop=True)
        logger.info(f"   Loaded {len(df_sample)} samples for testing.")
    except Exception as e:
        logger.error(f"FAILED to load CSV: {e}")
        return

    # 2. Config
    try:
        cfg = get_model_config(model_name)
    except Exception as e:
        logger.error(f"FAILED to load config for {model_name}: {e}")
        return

    # 3. Dataset & Encoder Initialization
    logger.info("2. Initializing Dataset...")
    try:
        dataset = DoubleTowerDataset(
            df=df_sample,
            drug_col=cfg.get("drug_col", "compound_id"),
            haplo_col=cfg.get("haplo_col", "genotype_id"),
            target_cols=cfg["targets"],
            multilabel_cols=list(MULTI_LABEL_COLS),
            preload_ram=False,
        )
        logger.info("   Dataset initialized successfully.")
    except Exception as e:
        logger.error(f"FAILED to initialize Dataset: {e}")
        return

    # 4. Check Single Item Loading
    logger.info("3. Checking __getitem__ (Single Sample)...")
    try:
        sample = dataset[0]
        drug_x = sample["drug_data"].x
        haplo_x = sample["haplo_data"].x
        logger.info(f"   Drug Feature Shape: {drug_x.shape}")
        logger.info(f"   Haplo Feature Shape: {haplo_x.shape}")

        # Verify Targets
        for t_col, t_val in sample["targets"].items():
            logger.info(f"   Target '{t_col}': {t_val} (Type: {t_val.dtype})")
    except Exception as e:
        logger.error(f"FAILED on __getitem__: {e}")
        return

    # 5. Check Batching (Collater)
    logger.info("4. Checking DataLoader & Collater...")
    try:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=DoubleTowerCollater(),
            num_workers=0,  # Main process for debugging
        )
        batch = next(iter(loader))

        logger.info(f"   Drug Batch Nodes: {batch['drug_batch'].x.shape}")
        logger.info(f"   Haplo Batch Nodes: {batch['haplo_batch'].x.shape}")
        logger.info("   Batching successful.")
    except Exception as e:
        logger.error(f"FAILED on Batching: {e}")
        return

    # 6. Model Initialization
    logger.info("5. Initializing Model...")
    try:
        # Detect dimensions from the loaded batch
        drug_dim = batch["drug_batch"].x.shape[1]
        haplo_dim = batch["haplo_batch"].x.shape[1]

        target_dims = {
            col: len(dataset.encoders[col].classes_) for col in cfg["targets"]
        }

        # Configs
        drug_config = {"num_features": drug_dim, "edge_dim": 0}
        haplo_config = {"num_features": haplo_dim, "edge_dim": 0}

        model = create_gnn_model(
            model_name=model_name,
            drug_config=drug_config,
            haplo_config=haplo_config,
            target_dims=target_dims,
            params=cfg["params"],
        ).to(device)
        logger.info("   Model created.")
    except Exception as e:
        logger.error(f"FAILED to create model: {e}")
        return

    # 7. Forward Pass
    logger.info("6. Running Forward Pass...")
    try:
        drug_in = batch["drug_batch"].to(device)
        haplo_in = batch["haplo_batch"].to(device)

        outputs = model(drug_in, haplo_in)

        for t_col, out_tensor in outputs.items():
            logger.info(f"   Output '{t_col}': shape {out_tensor.shape}")
            if torch.isnan(out_tensor).any():
                logger.warning(
                    f"   WARNING: NaN values detected in output for {t_col}!"
                )
    except Exception as e:
        logger.error(f"FAILED during Forward Pass: {e}")
        return

    # 8. Loss Calculation Check
    logger.info("7. Checking Loss Calculation...")
    try:
        total_loss = 0
        loss_fns = {}

        # Setup basic losses just for the check
        for col in cfg["targets"]:
            if col in MULTI_LABEL_COLS:
                loss_fns[col] = torch.nn.BCEWithLogitsLoss()
            else:
                loss_fns[col] = torch.nn.CrossEntropyLoss()

        for t_col, prediction in outputs.items():
            target = batch["targets"][t_col].to(device)

            # Type casting based on loss requirement
            if t_col in MULTI_LABEL_COLS:
                loss = loss_fns[t_col](prediction, target.float())
            else:
                loss = loss_fns[t_col](prediction, target.long())

            total_loss += loss
            logger.info(f"   Loss {t_col}: {loss.item():.4f}")

        # Backward check
        total_loss.backward()
        logger.info("   Backward pass successful. Gradients computed.")

    except Exception as e:
        logger.error(f"FAILED during Loss/Backward: {e}")
        return

    logger.info("--- SANITY CHECK PASSED SUCCESSFULLY ---")


if __name__ == "__main__":
    # Update these paths to match your local setup
    MODEL_NAME = "TwoTowerGAT"
    CSV_FILE = "./data/processed/training_data.tsv"  # Change to your actual data path

    run_sanity_check(MODEL_NAME, CSV_FILE)
