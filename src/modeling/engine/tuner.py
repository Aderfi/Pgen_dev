# Pharmagen - Optuna Hyperparameter Optimization
#
# Implements a comprehensive hyperparameter optimization pipeline using Optuna.
# Supports multi-objective optimization and clean architecture.

import json
import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import optuna
import torch
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Project Imports
from src.config.manager import get_model_config, MULTI_LABEL_COLS, DIRS
# Import the new loaders
from src.data.loaders import DoubleTowerDataset, DoubleTowerCollater
# Import the new Model Factory
from src.modeling.architectures.layers import create_gnn_model
from src.modeling.engine.trainer import PGenTrainer


logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

class OptunaTuner:
    """
    Orchestrator class for hyperparameter optimization.
    Encapsulates data, configuration, and Optuna lifecycle.
    """

    def __init__(
        self,
        model_name: str,
        csv_path: Union[str, Path],
        n_trials: int = 100,
        epochs: int = 75,
        patience: int = 15,
        random_seed: int = 711,
        device: Optional[torch.device] = None,
    ):
        self.model_name = model_name
        self.csv_path = Path(csv_path)
        self.n_trials = n_trials
        self.epochs = epochs
        self.patience = patience
        self.seed = random_seed
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Config
        self.config = get_model_config(model_name)
        self.feature_cols = [c.lower() for c in self.config["features"]] # e.g. ['drug_id', 'variant_id']
        self.target_cols = [t.lower() for t in self.config["targets"]]
        self.params = self.config["params"]
        self.optuna_space = self.config.get("params_optuna", {})

        # Paths to Graph Libraries (Assuming defined in DIRS or Config)
        self.drug_lib = DIRS.get("drugs_lib", Path("./data/library/drugs"))
        self.variant_lib = DIRS.get("gene_graphs", Path("./data/library/gene_graphs"))

        # Data Containers
        self.train_dataset: Optional[DoubleTowerDataset] = None
        self.val_dataset: Optional[DoubleTowerDataset] = None
        
        # Dimensions for Model Init
        self.drug_feat_dim = 0
        self.haplo_feat_dim = 0
        self.target_dims = {}

        self._prepare_data()

    def _prepare_data(self):
        """Initializes DoubleTowerDatasets and determines graph dimensions."""
        logger.info(f"Preparing Graph Data for {self.model_name}...")
        
        # 1. Load CSV
        df = pd.read_csv(self.csv_path, sep='\t' if str(self.csv_path).endswith('.tsv') else ',')
        
        # 2. Split
        # Helper stratify col usually generated during preprocessing
        stratify = df["_stratify"] if "_stratify" in df.columns else None
        train_df, val_df = train_test_split(
            df, test_size=0.2, stratify=stratify, random_state=self.seed
        )

        # 3. Initialize Datasets (This handles Target Encoding internally)
        # Note: We fit encoders on TRAIN and reuse them on VAL if the dataset supports it.
        # Ideally DoubleTowerDataset logic should support passing existing encoders.
        
        # Train Dataset
        self.train_dataset = DoubleTowerDataset(
            df=train_df,
            drug_col=self.config.get("drug_col", "compound_id"),
            haplo_col=self.config.get("haplo_col", "genotype_id"),
            target_cols=self.target_cols,
            multilabel_cols=list(MULTI_LABEL_COLS)
        )
        
        # Val Dataset (Reuse encoders from train to ensure consistency)
        self.val_dataset = DoubleTowerDataset(
            df=val_df,
            drug_col=self.config.get("drug_col", "compound_id"),
            haplo_col=self.config.get("haplo_col", "genotype_id"),
            target_cols=self.target_cols,
            multilabel_cols=list(MULTI_LABEL_COLS)
        )
        # Hack: Manually copy encoders to val dataset to ensure target mapping consistency
        self.val_dataset.encoders = self.train_dataset.encoders

        # 4. Determine Input Dimensions (Crucial for GATv2)
        # Peek at the first sample to get node feature sizes (x.shape[1])
        sample = self.train_dataset[0]
        self.drug_feat_dim = sample['drug_data'].x.shape[1]
        self.haplo_feat_dim = sample['haplo_data'].x.shape[1]
        
        # Determine Output Dimensions from Encoders
        for col in self.target_cols:
            enc = self.train_dataset.encoders[col]
            self.target_dims[col] = len(enc.classes_)

        logger.info(f"Graph Dims Detected -> Drug: {self.drug_feat_dim}, Haplo: {self.haplo_feat_dim}")
        logger.info(f"Target Dims: {self.target_dims}")


    # ==========================================================================
    # PARAMETER PARSING
    # ==========================================================================

    def _suggest_int(self, trial: optuna.Trial, name: str, args: List[Any]) -> int:
        # args: [low, high, step, log]
        low, high = args[0], args[1]
        step = args[2] if len(args) > 2 else 1
        log = args[3] if len(args) > 3 else False
        return trial.suggest_int(name, low, high, step=step, log=log)

    def _suggest_float(self, trial: optuna.Trial, name: str, args: Tuple[float, float]) -> float:
        # args: (low, high)
        low, high = args
        is_log = any(x in name for x in ["learning_rate", "weight_decay"])
        return trial.suggest_float(name, low, high, log=is_log)

    def _suggest_categorical(self, trial: optuna.Trial, name: str, choices: List[Any]) -> Any:
        return trial.suggest_categorical(name, choices)

    def _get_trial_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        params = self.config.get("params", {}).copy() # Start with defaults
        
        for name, space in self.optuna_space.items():
            try:
                if isinstance(space, list):
                    if not space: continue
                    
                    if space[0] == "int":
                        params[name] = self._suggest_int(trial, name, space[1:])
                    elif len(space) == 1:
                        params[name] = space[0] # Constant
                    else:
                        params[name] = self._suggest_categorical(trial, name, space)
                
                elif isinstance(space, tuple):
                    params[name] = self._suggest_float(trial, name, space)
                else:
                    params[name] = space
            
            except Exception as e:
                logger.error(f"Error suggesting param '{name}': {e}")
                raise
        return params

    # ==========================================================================
    # TRAINING LOOP (Objective)
    # ==========================================================================

    def objective(self, trial: optuna.Trial) -> float:
        # 1. Suggest Params
        params = self._get_trial_params(trial)
        
        # 2. DataLoaders with Custom Collater
        collater = DoubleTowerCollater()
        
        train_loader = DataLoader(
            self.train_dataset, 
            batch_size=params.get("batch_size", 32), 
            shuffle=True, 
            collate_fn=collater, # <--- CRITICAL: Batches graphs correctly
            num_workers=4, 
            pin_memory=True
        )
        val_loader = DataLoader(
            self.val_dataset, 
            batch_size=params.get("batch_size", 32), 
            shuffle=False, 
            collate_fn=collater,
            num_workers=4, 
            pin_memory=True
        )

        # 3. Model Creation
        # Adapt configs for create_gnn_model factory
        drug_config = {'num_features': self.drug_feat_dim, 'edge_dim': 0} # Update edge_dim if your .pt has edge attrs
        haplo_config = {'num_features': self.haplo_feat_dim, 'edge_dim': 0}

        model = create_gnn_model(
            model_name=self.model_name,
            drug_config=drug_config,
            haplo_config=haplo_config,
            target_dims=self.target_dims,
            params=params
        ).to(self.device)

        # 4. Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=params["learning_rate"], 
            weight_decay=params["weight_decay"]
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3)

        # 5. Train
        trainer = PGenTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=self.device,
            target_cols=self.target_cols,
            multi_label_cols=MULTI_LABEL_COLS,
            params=params
        )

        try:
            best_loss = trainer.fit(train_loader, val_loader, self.epochs, self.patience, trial)
            return best_loss
        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.error(f"Trial failed: {e}")
            raise e

    # ==========================================================================
    # EXECUTION & REPORTING
    # ==========================================================================

    def run(self):
        """Executes the full Optuna study."""
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M")
        study_name = f"OPT_{self.model_name}_{timestamp}"
        storage_url = f"sqlite:///{DIRS['reports']}/optuna_reports/study_DBs/{study_name}.db"

        logger.info(f"Starting study: {study_name}")

        sampler = TPESampler(seed=self.seed, multivariate=True)
        
        if self.use_multi_objective:
            directions = ["minimize", "minimize"]
            study = optuna.create_study(
                study_name=study_name,
                storage=storage_url,
                directions=directions,
                sampler=sampler,
                load_if_exists=True
            )
        else:
            pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5)
            study = optuna.create_study(
                study_name=study_name,
                storage=storage_url,
                direction="minimize",
                sampler=sampler,
                pruner=pruner,
                load_if_exists=True
            )

        # External Progress Bar
        with tqdm(total=self.n_trials, desc="Optuna Trials", colour="blue") as pbar:
            def progress_callback(study, trial):
                pbar.update(1)
                if study.best_trials:
                    best_val = study.best_trials[0].values[0]
                    pbar.set_postfix(best_loss=f"{best_val:.4f}")

            study.optimize(
                self.objective, 
                n_trials=self.n_trials, 
                callbacks=[progress_callback], 
                gc_after_trial=True
            )

        self._save_results(study, timestamp)
        return study

    def _save_results(self, study: optuna.Study, timestamp: str):
        """Generates JSON reports and plots."""
        logger.info("Generating reports...")
        
        # 1. Plots
        self._generate_plots(study, timestamp)

        # 2. JSON Report
        best_trials = study.best_trials
        base_name = f"report_{self.model_name}_{timestamp}"
        
        report_data = {
            "model": self.model_name,
            "best_trials": [
                {
                    "number": t.number,
                    "values": t.values,
                    "params": t.params,
                    "metrics": t.user_attrs
                } for t in best_trials
            ]
        }
        
        out_path = DIRS["reports"] / "optuna_reports" / f"{base_name}.json"
        with open(out_path, "w") as f:
            json.dump(report_data, f, indent=2)
            
        logger.info(f"Report saved to {out_path}")

    def _generate_plots(self, study, timestamp):
        """Safe wrapper for Optuna visualization."""
        try:
            from optuna.visualization.matplotlib import (
                plot_optimization_history,
                plot_param_importances,
            )
            
            base_path = DIRS["reports"] / "figures" / f"{self.model_name}_{timestamp}"
            
            plt.figure(figsize=(10, 6))
            plot_optimization_history(study)
            plt.tight_layout()
            plt.savefig(f"{base_path}_history.png")
            plt.close()

            if not self.use_multi_objective and len(study.trials) > 10:
                plt.figure(figsize=(10, 6))
                plot_param_importances(study)
                plt.tight_layout()
                plt.savefig(f"{base_path}_importance.png")
                plt.close()
                
        except Exception as e:
            logger.warning(f"Could not generate plots: {e}")


# ============================================================================
# ENTRY POINT
# ============================================================================

def run_optuna_study(
    model_name: str, 
    csv_path: Union[str, Path],
    n_trials: int = 100
):
    tuner = OptunaTuner(model_name, csv_path, n_trials=n_trials)
    study = tuner.run()
    
    print("\n" + "="*50)
    print(f"Best Trial Params: {study.best_trials[0].params}")
    print("="*50 + "\n")
