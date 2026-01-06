# tuner.py
# Pharmagen - Optuna Hyperparameter Optimization
#
# Implements a comprehensive hyperparameter optimization pipeline using Optuna.
# Supports multi-objective optimization and clean architecture.
# Copyright (C) 2025 Adrim Hamed Outmani

import datetime
import json
import logging
from pathlib import Path
from typing import Any, TypedDict, cast

import matplotlib.pyplot as plt
import optuna
import optuna.logging
import torch
import torch.multiprocessing as mp
from optuna.pruners import HyperbandPruner, MedianPruner, NopPruner, PatientPruner
from optuna.samplers import RandomSampler, TPESampler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm. auto import tqdm

# Project Imports
from src.config.manager import DIRS, get_model_config
from src.data.collator import DoubleTowerCollater
from src.data.datasets import DoubleTowerDataset
from src.interface.ui import ConsoleIO
from src. modeling.architectures. layers import create_gnn_model
from src.modeling.engine. trainer import PGenTrainer
from src.utils.exceptions import ConfigurationError, DataError, ModelError
from src.utils.io import DataLoaderUtils
from src.utils.memory import (
    MemoryMonitor,
    estimate_batch_memory_mb,
    estimate_model_memory_mb,
)
from src.utils.module_builder import LossFactory, OptimizerFactory
from src.utils.validation import ConfigValidator, DataValidator

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Constants
MIN_DATASET_SIZE = 1000
DEFAULT_MAX_BATCH_SIZE = 128
MEMORY_WARNING_THRESHOLD = 0.6  # 60% of available GPU memory


class PGenTuner:
    """
    Orchestrator for Optuna-based Hyperparameter Optimization.

    Responsibilities:
    - Data preparation and validation
    - Optuna study configuration and execution
    - Memory-efficient trial execution
    - Result reporting and visualization

    Features:
    - Automatic memory management with aggressive cleanup
    - Configurable samplers and pruners
    - Comprehensive validation and error handling
    - Progress reporting with detailed metrics
    - Multi-format result output (JSON, plots)

    Example:
        >>> tuner = PGenTuner(model_name="TwoTowerGAT", csv_path="data/train.tsv")
        >>> study = tuner.run_tuning(n_trials=50, n_jobs=4)
        >>> print(f"Best params: {study.best_params}")
    """

    def __init__(
        self,
        model_name: str,
        csv_path: str | Path,
        random_seed: int = 711,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
        sampler_type: str = "TPE",
        pruner_type:  str = "Hyperband",
    ):
        """Initialize PGenTuner with validation and configuration.

        Args:
            model_name:  Model configuration name from models.toml.
            csv_path: Path to training data (CSV/TSV).
            random_seed: Random seed for reproducibility.
            max_batch_size: Maximum allowed batch size (prevents OOM).
            sampler_type: Optuna sampler ("TPE", "Random", "Grid").
            pruner_type: Optuna pruner ("Hyperband", "Median", "None").

        Raises:
            ConfigurationError: If model config is invalid.
            DataError: If dataset is invalid or too small.
        """
        self.model_name = model_name
        self.csv_path = Path(csv_path)
        self.timestamp = datetime.datetime.now().strftime("%d_%m__%H_%M")
        self.study_name = f"OPT_{self.model_name}_{self.timestamp}"
        self.seed = random_seed
        self.max_batch_size = max_batch_size
        self.sampler_type = sampler_type
        self.pruner_type = pruner_type

        # Validación de parámetros y configuración
        try:
            self.cfg = get_model_config(model_name)
        except Exception as e:
            raise ConfigurationError(f"Failed to load config for '{model_name}': {e}") from e
        if "params_optuna" not in self.cfg:
            raise ConfigurationError(
                f"Model '{model_name}' missing 'params_optuna' configuration for tuning"
            )
        if not ConfigValidator.validate_optuna_params(self.cfg["params_optuna"]):
            raise ConfigurationError(
                f"Model '{model_name}' has invalid Optuna parameter definitions"
            )

        self.patience = self.cfg["params_optuna"].get("patience", 5)

        self.reports_dir = DIRS.get("reports", Path("./reports")) / "optuna_reports"
        self.figures_dir = DIRS.get("reports", Path("./reports")) / "figures"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        # Validación y carga de datos
        logger.info(f"📂 Loading data for tuning from {self.csv_path}...")
        try:
            self.full_df = DataLoaderUtils.load_dataframe(
                self.csv_path,
                cols=self.cfg["cols"],
                stratify_col=self.cfg. get("stratify_col", None),
            )
        except Exception as e:
            raise DataError(f"Failed to load data from {self.csv_path}: {e}") from e

        if len(self.full_df) < MIN_DATASET_SIZE:
            raise DataError(
                f"Dataset too small for Optuna tuning:  {len(self.full_df)} samples "
                f"(minimum: {MIN_DATASET_SIZE})"
            )

        # Validación de calidad de datos
        logger.info("🔍 Validating data quality...")
        missing_stats = DataValidator.check_missing_values(
            self.full_df,
            self.cfg["features"] + self.cfg["targets"],
            threshold=0.5,
        )
        if any(frac > 0.3 for frac in missing_stats. values()): # noqa
            logger.warning(
                f"⚠️ High missing values detected: {missing_stats}"
            )

        for target in self.cfg["targets"]:
            if target in self.full_df. columns:
                DataValidator.check_class_balance(
                    self.full_df,
                    target,
                    min_samples_per_class=10,
                )

        # Split data
        self.train_df, self.val_df = train_test_split(
            self.full_df,
            test_size=0.2,
            stratify=None,
            random_state=self.seed,
        )
        logger.info(
            f"✅ Tuning data ready: {len(self.train_df)} train, {len(self.val_df)} val"
        )

        # ✅ MEJORA 7: Log initial memory state
        MemoryMonitor.log_memory_stats("Initial memory - ")

    def _suggest_params(self, trial: optuna.Trial) -> dict[str, Any]:
        """
        Parse Optuna search space from configuration.

        Dynamically generates hyperparameter suggestions based on
        the 'params_optuna' section of the model configuration.

        Supported types:
        - categorical: ["categorical", val1, val2, ...]
        - int: ["int", min, max]
        - float: ["float", min, max]
        - log: ["log", min, max] (log-scale float)

        Args:
            trial: Optuna trial object.

        Returns:
            Dictionary of suggested hyperparameters.

        Raises:
            ConfigurationError: If parameter definition is invalid.
        """
        suggestions = {}
        optuna_conf = self.cfg.get("params_optuna", {})

        for param_name, args in optuna_conf.items():
            # Skip non-searchable config keys
            if param_name in {"patience", "epochs"}:
                continue

            if not isinstance(args, list) or len(args) == 0:
                continue

            try:
                ptype = args[0]
                if ptype == "categorical":
                    suggestions[param_name] = trial.suggest_categorical(
                        param_name, args[1:]
                    )
                elif ptype == "int":
                    suggestions[param_name] = trial.suggest_int(
                        param_name, args[1], args[2]
                    )
                elif ptype == "float":
                    suggestions[param_name] = trial.suggest_float(
                        param_name, args[1], args[2]
                    )
                elif ptype == "log":
                    suggestions[param_name] = trial.suggest_float(
                        param_name, args[1], args[2], log=True
                    )
                else:
                    logger.warning(
                        f"⚠️ Unknown param type '{ptype}' for '{param_name}', skipping"
                    )
            except Exception as e:
                raise ConfigurationError(
                    f"Invalid Optuna param '{param_name}': {e}"
                ) from e

        return suggestions

    def objective(self, trial: optuna. Trial) -> float: # noqa
        """
        Optuna objective function for single trial.

        Executes complete training cycle with suggested hyperparameters:
        1. Clear memory and log state
        2. Suggest hyperparameters
        3. Create fresh datasets
        4. Infer dimensions
        5. Create model, optimizer, trainer
        6. Train and validate
        7. Cleanup resources

        Args:
            trial:  Optuna trial object.

        Returns:
            Best validation loss achieved.

        Raises:
            optuna. TrialPruned: If trial should be terminated early.
        """
        # Clear memory before starting trial
        MemoryMonitor. clear_memory(aggressive=True)
        MemoryMonitor.log_memory_stats(f"🔬 Trial {trial.number} start - ")

        try:
            # 1. Hyperparameter suggestion
            params = self._suggest_params(trial)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            batch_size = int(params.get("batch_size", 64))

            if batch_size > self.max_batch_size:
                logger.warning(
                    f"⚠️ Trial {trial.number}: Batch size {batch_size} exceeds "
                    f"max {self.max_batch_size}, capping"
                )
                batch_size = self.max_batch_size

                # Track capping for analysis
                trial.set_user_attr("batch_size_capped", True)
                trial.set_user_attr("original_batch_size", params.get("batch_size"))

            epochs = int(self.cfg.get("params_optuna", {}).get("epochs", 50))

            # 2. Memory estimation with preventive check
            estimated_batch_mem = estimate_batch_memory_mb(
                batch_size=batch_size,
                avg_nodes_per_graph=50,
                node_features=25,
                num_graphs=2,
            )
            logger.debug(f"💾 Trial {trial.number}:  Estimated batch memory={estimated_batch_mem:.1f}MB")

            # Preventive OOM check
            if torch.cuda.is_available():
                available_mem = torch.cuda.get_device_properties(0).total_memory / 1024**2
                mem_ratio = estimated_batch_mem / available_mem

                if mem_ratio > MEMORY_WARNING_THRESHOLD:
                    logger.warning(
                        f"⚠️ Trial {trial.number}: High memory usage expected "
                        f"({estimated_batch_mem:.0f}MB / {available_mem:.0f}MB = {mem_ratio:.1%})"
                    )
                    trial.set_user_attr("high_memory_risk", True)

                    # Optional: Aggressively prune high-memory trials
                    if mem_ratio > 0.85:  # noqa 85% threshold 
                        logger.error(
                            f"❌ Trial {trial.number}:  Batch size too large, pruning preemptively"
                        )
                        raise optuna. TrialPruned("Batch size exceeds memory limits")

            # 3. Dataset construction (always fresh per trial)
            train_dataset = DoubleTowerDataset(
                df=self.train_df,
                drug_col=self.cfg["features"][0],
                haplo_col="haplo_key",
                target_cols=self.cfg["targets"],
                multilabel_cols=self.cfg.get("multi_label_cols", []),
                preload_ram=False,  # CRITICAL: Always False during Optuna
            )

            val_dataset = DoubleTowerDataset(
                df=self.val_df,
                drug_col=self.cfg["features"][0],
                haplo_col="haplo_key",
                target_cols=self.cfg["targets"],
                multilabel_cols=self.cfg.get("multi_label_cols", []),
                encoders=train_dataset.encoders,  # CRITICAL: Share encoders
                preload_ram=False,
            )

            # 4. Dimension inference
            try:
                sample = train_dataset[0]
                drug_dim = sample["drug_data"].x.shape[1]
                haplo_dim = sample["haplo_data"].x.shape[1]
            except Exception as e:
                logger.error(f"❌ Trial {trial.number}:  Dimension inference failed: {e}")
                raise DataError(f"Failed to infer dimensions from dataset:  {e}") from e

            target_dims = {
                col: len(train_dataset.encoders[col].classes_)
                for col in self.cfg["targets"]
                if col in train_dataset.encoders
            }

            # Fallback for missing target dimensions
            for t in self.cfg["targets"]:
                if t not in target_dims:
                    logger.warning(f"⚠️ Target '{t}' not in encoders, defaulting to dim=1")
                    target_dims[t] = 1

            # 5. DataLoaders
            collater = DoubleTowerCollater()
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collater,
                num_workers=0,
                pin_memory=True if torch.cuda.is_available() else False,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collater,
                num_workers=0,
                pin_memory=True if torch.cuda.is_available() else False,
            )

            # 6. Model instantiation
            model = create_gnn_model(
                model_name=self.model_name,
                drug_config={
                    "num_features": drug_dim,
                    "edge_dim": self.cfg.get("drug_edge_dim", 0),
                },
                haplo_config={
                    "num_features": haplo_dim,
                    "edge_dim": self.cfg.get("haplo_edge_dim", 0),
                },
                target_dims=target_dims,
                params=params,
            ).to(device)

            # Enhanced logging
            num_params = sum(p.numel() for p in model.parameters())
            model_mem = estimate_model_memory_mb(num_params)
            logger.debug(
                f"🧠 Trial {trial.number}:  Model params={num_params: ,}, mem~{model_mem:.1f}MB"
            )

            # 7. Loss & Uncertainty Setup
            uncertainty_net = LossFactory.create_uncertainty_wrapper(
                tasks=self.cfg["targets"], device=device
            )

            # 8. Optimizer & Scheduler
            optimizer = OptimizerFactory.create(
                model=model,
                params=params,
                uncertainty_module=uncertainty_net,
            )

            scheduler = torch.optim.lr_scheduler. ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5
            )

            # 9. Trainer Setup
            trainer = PGenTrainer(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                target_cols=self.cfg["targets"],
                multi_label_cols=set(self.cfg.get("multi_label_cols", [])),
                params=params,
                uncertainty_module=uncertainty_net,
                from_optuna=True,
            )

            # 10. Training with memory monitoring
            MemoryMonitor.log_memory_stats(f"🏋️ Trial {trial.number} before training - ")
            result = trainer.fit(
                train_loader,
                val_loader,
                epochs=epochs,
                patience=self.patience,
                trial=trial,
            )

            return result

        except optuna.TrialPruned:
            raise
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"❌ Trial {trial.number}: OOM error: {e}")
                MemoryMonitor.clear_memory(device=device, aggressive=True) # type: ignore
                raise optuna.TrialPruned("Out of memory")
            logger.error(f"❌ Trial {trial.number}: RuntimeError: {e}")
            raise optuna.TrialPruned(f"RuntimeError: {e}")
        except Exception as e:
            logger. error(f"❌ Trial {trial.number}: Unexpected error: {e}", exc_info=True)
            raise optuna.TrialPruned(f"Unexpected error: {e}")
        finally:
            # Enhanced cleanup with list comprehension
            local_vars_to_cleanup = [
                'train_loader', 'val_loader', 'train_dataset', 'val_dataset',
                'model', 'optimizer', 'scheduler', 'trainer', 'uncertainty_net'
            ]
            for var_name in local_vars_to_cleanup:
                if var_name in locals():
                    del locals()[var_name]

            # Aggressive memory cleanup
            MemoryMonitor.clear_memory(aggressive=True)
            MemoryMonitor.log_memory_stats(f"🧹 Trial {trial.number} cleanup - ")

    def _create_sampler(self) -> optuna.samplers.BaseSampler:
        """
        Returns: Configured Optuna sampler.
        """
        if self.sampler_type == "TPE":
            return TPESampler(seed=self.seed, multivariate=True)
        elif self.sampler_type == "Random":
            return RandomSampler(seed=self.seed)
        elif self.sampler_type == "Grid":
            logger.warning("Grid sampler requires explicit search space, using TPE")
            return TPESampler(seed=self.seed, multivariate=True)
        else:
            logger.warning(f"Unknown sampler '{self.sampler_type}', using TPE")
            return TPESampler(seed=self.seed, multivariate=True)

    def _create_pruner(self) -> optuna.pruners.BasePruner:
        """
        Returns:
            Configured Optuna pruner.
        """
        if self.pruner_type == "Hyperband":
            base = HyperbandPruner(
                min_resource=5,
                max_resource=50,
                reduction_factor=3
            )
            return PatientPruner(base, patience=5)
        elif self.pruner_type == "Median":
            return MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        elif self.pruner_type == "None":
            return NopPruner()
        else:
            logger.warning(f"Unknown pruner '{self.pruner_type}', using Hyperband")
            # Default to Hyperband
            base = HyperbandPruner(min_resource=5, max_resource=50, reduction_factor=3)
            return PatientPruner(base, patience=5)

    def run_tuning(self, n_trials: int = 50, n_jobs: int | None = None) -> optuna.Study:
        """
        Execute complete Optuna study.

        Args:
            n_trials: Number of trials to run.
            n_jobs: Number of parallel workers (None = auto-detect).

        Returns:
            Completed Optuna study object.
        """
        if n_jobs is None:
            n_jobs = 4 if torch.cuda.is_available() else 1

        logger.info(f"🚀 Starting Optuna Study:  {self.study_name} w/ n_jobs={n_jobs}")
        logger.info(f"📊 Sampler: {self.sampler_type}, Pruner: {self.pruner_type}")

        # ✅ MEJORA 17: Use factory methods for sampler/pruner
        sampler = self._create_sampler()
        pruner = self._create_pruner()

        # Setup study storage
        storage_url = self.reports_dir / "study_DBs" / f"{self.study_name}.db"
        storage_url.parent.mkdir(parents=True, exist_ok=True)
        db_path = f"sqlite:///{storage_url.resolve()}"

        study = optuna.create_study(
            study_name=self.study_name,
            storage=db_path,
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )

        if n_jobs > 1:
            logger.info("⚙️ Running in parallel mode.  TQDM bar disabled.")
            study.optimize(
                self. objective,
                n_trials=n_trials,
                n_jobs=n_jobs,
                gc_after_trial=True,
            )
        else:
            # Single-job mode with progress bar
            with tqdm(total=n_trials, desc="Optuna Trials", colour="blue") as pbar:
                def callback(study, trial):
                    pbar.update(1)
                    if study.best_trial:
                        best_trial = study.best_trial
                        best_params_str = ", ".join(
                            f"{k}={v}" for k, v in list(best_trial.params.items())[:3]
                        )
                        pbar.set_postfix({
                            "Trial": best_trial.number,
                            "Best":  f"{best_trial.value:.4f}",
                            "Params": best_params_str,
                        })

                study.optimize(
                    self.objective,
                    n_trials=n_trials,
                    callbacks=[callback],
                    gc_after_trial=True,
                )

        self._save_results(study)
        return study

    def _save_results(self, study: optuna. Study):
        """
        Generate comprehensive reports and visualizations.
        """
        best_trial = study.best_trial

        # All Trials with valid results
        completed_trials = [
            t for t in study.trials
            if t.state is optuna.trial.TrialState.COMPLETE
        ]

        completed_trials.sort(key=lambda t: t.value)  # type: ignore
        top_5 = completed_trials[:5]

        # ✅ MEJORA 19: More comprehensive report
        report: dict[str, Any] = {
            "model":  self.model_name,
            "study_name": self.study_name,
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
                    t for t in study. trials
                    if t.state == optuna.trial.TrialState.PRUNED
                ]),
                "n_failed": len([
                    t for t in study.trials
                    if t.state == optuna.trial.TrialState.FAIL
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

        # Save JSON report
        json_path = self.reports_dir / f"report_{self.study_name}. json"
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"📊 Report saved to {json_path}")

        # ✅ MEJORA 20: Console summary with ConsoleIO
        ConsoleIO.print_header("Optuna Study Results")
        ConsoleIO.print_success(f"Best Trial: #{best_trial.number}")
        ConsoleIO.print_info(f"Best Loss: {best_trial.value:.4f}")
        ConsoleIO.print_info(
            f"Completed Trials: {report['statistics']['n_completed']}"
            f"/{report['statistics']['n_trials']}"
        )
        ConsoleIO.print_info(f"Pruned:  {report['statistics']['n_pruned']}")

        if report['statistics']['n_failed'] > 0:
            ConsoleIO. print_warning(f"Failed:  {report['statistics']['n_failed']}")

        # Visualizations
        try:
            from optuna.visualization. matplotlib import (
                plot_optimization_history,
                plot_param_importances,
            )

            # Optimization history
            plt.figure(figsize=(10, 6))
            plot_optimization_history(study)
            plt.title(f"Optimization History - {self.model_name}")
            plt.tight_layout()
            history_path = self.figures_dir / f"{self.study_name}_history.png"
            plt.savefig(history_path, dpi=150)
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
                logger. info(f"📊 Importance plot saved to {importance_path}")

            logger.info("✅ Optimization plots generated successfully")

        except ImportError:
            logger.warning(
                "⚠️ Matplotlib not available in optuna. visualization, skipping plots"
            )
        except Exception as e:
            logger.warning(f"⚠️ Plot generation failed: {e}")


def run_optuna_study(
    model_name: str,
    csv_path: str | Path,
    n_trials: int = 50,
    epochs: int | None = None,
    sampler:  str = "TPE",
    pruner: str = "Hyperband",
) -> optuna.Study:
    """
    Args:
        model_name: Model configuration name.
        csv_path: Path to training data.
        n_trials: Number of Optuna trials.
        epochs:  Epochs per trial (optional, uses config default).
        sampler: Sampler type ("TPE", "Random", "Grid").
        pruner: Pruner type ("Hyperband", "Median", "None").

    Returns:
        Completed Optuna study.

    Example:
        >>> study = run_optuna_study("TwoTowerGAT", "data/train.tsv", n_trials=100)
        >>> print(study.best_params)
    """

    current_method = mp.get_start_method(allow_none=True)
    if current_method != 'spawn':

        if current_method is not None:
            logger.warning(
                f"⚠️ Overriding multiprocessing:  '{current_method}' → 'spawn' "
                f"(required for CUDA safety)"
            )

        try:
            mp.set_start_method('spawn', force=True)
            logger.debug("✅ Multiprocessing:  'spawn' (CUDA-safe)")
        except RuntimeError as e:
            logger.error(f"❌ Failed to configure multiprocessing: {e}")
            if "cuda" in str(e).lower() or torch.cuda.is_available():
                raise  # Re-raise si CUDA está disponible (crítico)
            else:
                logger.warning("   Continuing (CPU-only mode)")

    # ✅ MEJORA 22: Use ConsoleIO for better output
    ConsoleIO.print_header("Optuna Hyperparameter Optimization")
    ConsoleIO.print_info(f"Model: {model_name}")
    ConsoleIO.print_info(f"Data: {csv_path}")
    ConsoleIO.print_info(f"Trials: {n_trials}")
    ConsoleIO.print_info(f"Sampler: {sampler}")
    ConsoleIO.print_info(f"Pruner: {pruner}")
    ConsoleIO.print_divider("=")

    # Initialize and run tuner
    tuner = PGenTuner(
        model_name=model_name,
        csv_path=csv_path,
        sampler_type=sampler,
        pruner_type=pruner,
    )
    study = tuner.run_tuning(n_trials=n_trials, n_jobs=4)

    # ✅ MEJORA 23: Better final summary
    ConsoleIO.print_divider("=")
    ConsoleIO.print_success("🎉 Optuna Study Completed!")
    ConsoleIO.print_info(f"Best Loss: {study.best_value:.4f}")
    ConsoleIO.print_info("Best Parameters:")
    for param, value in study.best_params.items():
        print(f"  • {param}: {value}")
    ConsoleIO.print_divider("=")

    return study

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2: # noqa
        print("Usage: python tuner.py <model_name> [csv_path] [n_trials]")
        sys.exit(1)

    model = sys.argv[1]
    data = sys.argv[2] if len(sys.argv) > 2 else "train_data/train_data.tsv" # noqa
    trials = int(sys.argv[3]) if len(sys.argv) > 3 else 50 # noqa

    run_optuna_study(model, data, trials)
