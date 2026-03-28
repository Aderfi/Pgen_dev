# tuner.py
# Pharmagen - Optuna Hyperparameter Optimization
#
# Implements a comprehensive hyperparameter optimization pipeline using Optuna.
# Supports multi-objective optimization and clean architecture.
# Copyright (C) 2025 Adrim Hamed Outmani

import datetime
import gc
import json
import logging
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import optuna
import optuna.logging
import torch
import torch.multiprocessing as mp
from optuna.pruners import HyperbandPruner, MedianPruner, NopPruner, PatientPruner
from optuna.samplers import RandomSampler, TPESampler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Project Imports
from src.config.manager import DIRS, get_model_config
from src.data.collator import DoubleTowerCollater
from src.data.datasets import DoubleTowerDataset
from src.interface.ui import ConsoleIO
from src.modeling.architectures.layers import create_gnn_model
from src.modeling.engine.trainer import PGenTrainer
from src.utils.exceptions import ConfigurationError, DataError
from src.utils.io import DataLoaderUtils
from src.utils.memory import estimate_batch_memory_mb
from src.utils.module_builder import LossFactory, OptimizerFactory
from src.utils.validation import ConfigValidator, DataValidator

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Constants
MIN_DATASET_SIZE = 1000
DEFAULT_MAX_BATCH_SIZE = 128
N_JOBS = 1

@contextmanager
def trial_context(trial_number: int, device: torch.device) -> Generator[None, None, None]:
    """
    Context Manager to handle Trial setup, error handling (OOM), and cleanup.
    """
    # 1. Setup (Clean state)
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    try:
        yield
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error(f"❌ Trial {trial_number}: OOM Error. Pruning.")
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            raise optuna.TrialPruned("CUDA OOM")
        logger.error(f"❌ Trial {trial_number}: Runtime Error: {e}")
        raise e
    except Exception as e:
        logger.error(f"❌ Trial {trial_number}: Unexpected Error: {e}")
        raise e
    finally:
        # 3. Teardown (Aggressive cleanup)
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

class PGenTuner:
    """
    Orchestrator for Optuna-based Hyperparameter Optimization.
    """

    def __init__(
        self,
        model_name: str,
        csv_path: str | Path,
        random_seed: int = 711,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
        sampler_type: str = "TPE",
        pruner_type: str = "Hyperband",
    ):
        self.model_name = model_name
        self.csv_path = Path(csv_path)
        self.seed = random_seed
        self.max_batch_size = max_batch_size
        self.sampler_type = sampler_type
        self.pruner_type = pruner_type

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.timestamp = datetime.datetime.now().strftime("%d_%m__%H_%M")
        self.study_name = f"OPT_{self.model_name}_{self.timestamp}"

        # Artifact Paths
        self.reports_dir = DIRS.get("reports", Path("./reports")) / "optuna"
        self.figures_dir = DIRS.get("reports", Path("./reports")) / "figures"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        # Config Loading & Validation
        self._load_and_validate_config()

        # Components Reuse
        self.collater = DoubleTowerCollater() # Stateless, reuse across trials

        # Data Loading
        self._initialize_static_data()

    def _load_and_validate_config(self):
        """Validates configuration independently (SRP)."""
        try:
            self.cfg = get_model_config(self.model_name)
        except Exception as e:
            raise ConfigurationError(f"Failed to load config: {e}") from e

        if "params_optuna" not in self.cfg:
            raise ConfigurationError(f"Model '{self.model_name}' missing 'params_optuna'")

        if not ConfigValidator.validate_optuna_params(self.cfg["params_optuna"]):
            raise ConfigurationError("Invalid Optuna parameter definitions")

        self.patience = self.cfg["params_optuna"].get("patience", 5)

    def _initialize_static_data(self):
        """Loads data once (Static Data Pattern)."""
        logger.info(f"📂 Loading static data from {self.csv_path}...")

        full_df = DataLoaderUtils.load_dataframe(
            self.csv_path,
            cols=self.cfg["cols"],
            stratify_col=self.cfg.get("stratify_col", None),
        )

        if len(full_df) < MIN_DATASET_SIZE:
            raise DataError(f"Dataset too small: {len(full_df)}")

        # Validation
        DataValidator.check_missing_values(
            df=full_df,
            columns=[c for c in (self.cfg["features"] + self.cfg["targets"]) if c in full_df.columns],
            threshold=0.5
        )

        stratify_labels = full_df["_stratify"] if "_stratify" in full_df.columns else None

        train_df, val_df = train_test_split(
            full_df,
            test_size=0.2,
            stratify=stratify_labels,
            random_state=self.seed,
        )

        logger.info("🛠️ Instantiating Datasets (CPU Mode)...")
        # Note: preload_ram=False is enforced for safety based on previous optimizations
        self.train_dataset = DoubleTowerDataset(
            df=train_df,
            drug_col=self.cfg["features"][0],
            geno_col="geno_key",
            target_cols=self.cfg["targets"],
            multilabel_cols=self.cfg.get("multi_label_cols", []),
            preload_ram=False,
        )

        self.val_dataset = DoubleTowerDataset(
            df=val_df,
            drug_col=self.cfg["features"][0],
            geno_col="geno_key",
            target_cols=self.cfg["targets"],
            multilabel_cols=self.cfg.get("multi_label_cols", []),
            encoders=self.train_dataset.encoders,
            preload_ram=False,
        )

        self._infer_dimensions()

        # Cleanup
        del full_df, train_df, val_df, stratify_labels
        gc.collect()

    def _infer_dimensions(self):
        """Infers input dimensions from a single sample."""
        try:
            sample = self.train_dataset[0]
            self.drug_dim = sample["drug_data"].x.shape[1]
            self.geno_dim = sample["geno_data"].x.shape[1]
            self.target_dims = {
                col: len(self.train_dataset.encoders[col].classes_)
                if col in self.train_dataset.encoders else 1
                for col in self.cfg["targets"]
            }
            logger.info(f"📏 Dimensions: Drug={self.drug_dim}, Geno={self.geno_dim}")
        except Exception as e:
            raise DataError(f"Failed to infer dimensions: {e}") from e

    def _suggest_params(self, trial: optuna.Trial) -> dict[str, Any]:
        """Maps config definitions to Optuna trial suggestions."""
        suggestions = {}
        optuna_conf = self.cfg.get("params_optuna", {})

        for param_name, args in optuna_conf.items():
            if param_name in {"patience", "epochs"} or not isinstance(args, list):
                continue
            try:
                ptype, *vals = args
                if ptype == "categorical":
                    suggestions[param_name] = trial.suggest_categorical(param_name, vals)
                elif ptype == "int":
                    suggestions[param_name] = trial.suggest_int(param_name, vals[0], vals[1])
                elif ptype == "float":
                    suggestions[param_name] = trial.suggest_float(param_name, vals[0], vals[1])
                elif ptype == "log":
                    suggestions[param_name] = trial.suggest_float(param_name, vals[0], vals[1], log=True)
            except Exception:
                logger.warning(f"Skipping invalid param definition: {param_name}")

        return suggestions

    def _build_pipeline(self, params: dict) -> tuple[torch.nn.Module, PGenTrainer]:
        """Factory method to build the training pipeline components."""

        model = create_gnn_model(
            model_name=self.model_name,
            drug_config={"num_features": self.drug_dim, "edge_dim": self.cfg.get("drug_edge_dim", 0)},
            geno_config={"num_features": self.geno_dim, "edge_dim": self.cfg.get("geno_edge_dim", 0)},
            target_dims=self.target_dims,
            params=params,
        ).to(self.device)

        uncertainty_net = LossFactory.create_uncertainty_wrapper(
            tasks=self.cfg["targets"], device=self.device
        )

        optimizer = OptimizerFactory.create(
            model=model, params=params, uncertainty_module=uncertainty_net
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        trainer = PGenTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=self.device,
            target_cols=self.cfg["targets"],
            multi_label_cols=set(self.cfg.get("multi_label_cols", [])),
            params=params,
            uncertainty_module=uncertainty_net,
            from_optuna=True,
        )

        return model, trainer

    def objective(self, trial: optuna.Trial) -> float:
        """Atomic Unit of Work (The Trial)."""

        #with trial_context(trial.number, self.device):
        # 1. Hyperparameters
        params = self._suggest_params(trial)
        batch_size = min(int(params.get("batch_size", 64)), self.max_batch_size)
        epochs = int(self.cfg.get("params_optuna", {}).get("epochs", 50))

        # 2. DataLoaders
        train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=self.collater, num_workers=0, pin_memory=True
        )
        val_loader = DataLoader(
            self.val_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=self.collater, num_workers=0, pin_memory=True
        )

        # 3. Build & Train
        _, trainer = self._build_pipeline(params)

        result = trainer.fit(
            train_loader, val_loader, epochs=epochs, patience=self.patience, trial=trial
        )

        return result

    def tune(self, n_trials: int, n_jobs: int | None = 1) -> optuna.Study:
        """Executes the optimization study."""

        # Sampler/Pruner Factory
        sampler = (TPESampler(seed=self.seed, multivariate=True)
                   if self.sampler_type == "TPE" else RandomSampler(seed=self.seed))

        pruner = (PatientPruner(HyperbandPruner(), patience=5)
                  if self.pruner_type == "Hyperband" else MedianPruner())

        # Storage
        storage_url = f"sqlite:///{self.reports_dir.parent}/database/{self.study_name}.db"
        Path(storage_url.replace("sqlite:///", "")).parent.mkdir(parents=True, exist_ok=True)

        study = optuna.create_study(
            study_name=self.study_name,
            storage=storage_url,
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )

        logger.info(f"🚀 Starting Study: {self.study_name}")
        if n_jobs is None:
            n_jobs = 1
        if n_jobs > 1:
            study.optimize(
                self.objective,
                n_trials=n_trials,
                n_jobs=n_jobs,
                gc_after_trial=True
            )
        elif n_jobs == 1:
            # Progress Bar Callback
            with tqdm(total=n_trials, desc="Trials", colour="blue") as pbar:
                def callback(study, trial):
                    pbar.update(1)
                    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
                    if completed:
                        best = study.best_trial
                        pbar.set_postfix({"Best": f"{best.value:.4f}", "Trial": trial.number})
                    else:
                        pbar.set_postfix({"Status": trial.state.name})

                study.optimize(
                    self.objective,
                    n_trials=n_trials,
                    n_jobs=1,
                    callbacks=[callback],
                    gc_after_trial=True
                )

        self._save_results(study)
        return study

    def _save_results(self, study: optuna.Study):
        """Generates reports and plots."""
        # Check for completion
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed_trials or len(completed_trials) < 1:
            logger.warning("No trials completed successfully. Skipping report.")
            return

        is_minimize = study.direction == optuna.study.StudyDirection.MINIMIZE

        best_trial = study.best_trial
        completed_trials.sort(key=lambda t: t.value, reverse=not is_minimize) # type: ignore ; Linter bug
        top_5 = completed_trials[:5]

        # JSON Report
        report: dict[str, Any] = {
            "model":  self.model_name,
            "targets": self.cfg["targets"],
            "direction": study.direction.name,
            "best_trial": {
                "number": best_trial.number,
                "value":  best_trial.value,
                "params": best_trial.params,
                "datetime": (
                    best_trial.datetime_start.isoformat()
                    if best_trial.datetime_start
                    else None
                ),
            },
            "statistics": {
                "n_trials": len(study.trials),
                "n_completed": len(completed_trials),
                "n_pruned": len([
                    t for t in study.trials
                    if t.state == optuna.trial.TrialState.PRUNED
                ]),
            },
            "top_5_trials": [
                {"trial":  t.number, "value": t.value, "params": t.params}
                for t in top_5
            ],
            "configuration": {
                "sampler": self.sampler_type,
                "pruner": self.pruner_type,
                "max_batch_size": self.max_batch_size,
                "random_seed": self.seed,
            },
            "datetime": self.timestamp,
        }

        json_path = self.reports_dir / "reports" / f"optuna_{self.study_name.replace('OPT_', '')}.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)

        ConsoleIO.print_success(f"Report saved: {json_path}")

        # Plots
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from optuna.visualization.matplotlib import (
                plot_optimization_history,
                plot_param_importances,
            )

            # Optimization history
            plt.figure(figsize=(10, 6))
            plot_optimization_history(study)
            plt.title(f"Optimization History - {self.model_name}")
            plt.tight_layout()
            history_path = self.figures_dir / f"{self.study_name}_history.png"
            plt.savefig(history_path, dpi=300)
            plt.close()
            logger.info(f"📈 History plot saved to {history_path}")

            # Parameter importance (only if enough completed trials)
            if len(completed_trials) > 1:
                plt.figure(figsize=(10, 6))
                plot_param_importances(study)
                plt.title(f"Parameter Importance - {self.model_name}")
                plt.tight_layout()
                importance_path = self.figures_dir / f"{self.study_name}_importance.png"
                plt.savefig(importance_path, dpi=150)
                plt.close()
                logger.info(f"📊 Importance plot saved to {importance_path}")

            ConsoleIO.print_success("✅ Optimization plots generated successfully")
            ConsoleIO.print_info(f"Plots saved in: {self.figures_dir}")

        except ImportError:
            logger.warning(
                "⚠️ Matplotlib not available in optuna. visualization, skipping plots"
            )
        except Exception as e:
            ConsoleIO.print_error(f"Plot generation failed: {e}")

# ==============================================================================
# ENTRY POINT
# ==============================================================================

def run_optuna_study(
    model_name: str,
    csv_path: str | Path,
    n_trials: int = 50,
    epochs: int | None = None,
    sampler: str = "TPE",
    pruner: str = "Hyperband",
) -> optuna.Study:
    """Wrapper function to maintain compatibility with CLI/Main."""

    # Force Spawn for CUDA safety
    if mp.get_start_method(allow_none=True) != 'spawn':
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

    tuner = PGenTuner(
        model_name=model_name,
        csv_path=csv_path,
        sampler_type=sampler,
        pruner_type=pruner,
    )

    # Runtime Override
    if epochs:
        if "params_optuna" in tuner.cfg:
            tuner.cfg["params_optuna"]["epochs"] = epochs

    # Execute
    return tuner.tune(n_trials=n_trials, n_jobs=N_JOBS)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2: # noqa
        print("Usage: python tuner.py <model_name> [csv_path] [n_trials]")
        sys.exit(1)

    run_optuna_study(
        model_name=sys.argv[1],
        csv_path=sys.argv[2] if len(sys.argv) > 2 else "train_data/train_data.tsv", # noqa
        n_trials=int(sys.argv[3]) if len(sys.argv) > 3 else 50 # noqa
    )

