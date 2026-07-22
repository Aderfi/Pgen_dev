# Pharmagen - Optuna Hyperparameter Optimization
#
# Implements a comprehensive hyperparameter optimization pipeline using Optuna.
# Supports multi-objective optimization and clean architecture.
# Copyright (C) 2025 Adrim Hamed Outmani

from __future__ import annotations

import datetime
import gc
import json
import logging
from pathlib import Path
from typing import Any

import optuna
import optuna.logging
import torch
import torch.multiprocessing as mp
from optuna.pruners import HyperbandPruner, MedianPruner, PatientPruner
from optuna.samplers import RandomSampler, TPESampler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.config import get_axes_config, get_model_config, get_settings
from src.core import ConfigurationError, ConfigValidator
from src.data.collator import DoubleTowerCollater
from src.interface.ui import ConsoleIO
from src.model.architectures.assembly import infer_axis_specs
from src.model.engine.base import (
    build_gnn_model,
    build_two_tower_datasets,
    extract_tower_dims,
    infer_dimensions,
    load_and_clean_data,
    resolve_device,
    stratified_split,
)
from src.model.factories import OptimizerFactory
from src.model.losses import CompositionalLabelLoss, MultiTaskLoss
from src.model.training.optuna_trainer import OptunaTrialTrainer

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

MIN_DATASET_SIZE = 1000
DEFAULT_MAX_BATCH_SIZE = 128
N_JOBS = 1


class PGenTuner:
    """Orchestrator for Optuna-based Hyperparameter Optimization."""

    def __init__(
        self,
        model_name: str,
        csv_path: str | Path,
        random_seed: int = 711,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
        sampler_type: str = "TPE",
        pruner_type: str = "Hyperband",
        epochs_override: int | None = None,
    ):
        self.model_name = model_name
        self.csv_path = Path(csv_path)
        self.seed = random_seed
        self.max_batch_size = max_batch_size
        self.sampler_type = sampler_type
        self.pruner_type = pruner_type
        self.epochs_override = epochs_override

        self.device = resolve_device()
        self.timestamp = datetime.datetime.now().strftime("%d_%m__%H_%M")
        self.study_name = f"OPT_{self.model_name}_{self.timestamp}"

        paths = get_settings().paths
        self.reports_dir = paths.reports / "optuna"
        self.figures_dir = paths.reports / "figures"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        self._load_and_validate_config()

        # Stateless components reused across trials.
        self.collater = DoubleTowerCollater()
        self.dims = extract_tower_dims(self.cfg)

        self._initialize_static_data()

    def _load_and_validate_config(self) -> None:
        try:
            self.cfg = get_model_config(self.model_name)
        except Exception as e:
            raise ConfigurationError(f"Failed to load config: {e}") from e

        if not self.cfg.optuna:
            raise ConfigurationError(
                f"Model '{self.model_name}' missing 'optuna' search space"
            )

        if not ConfigValidator.validate_optuna_params(self.cfg.optuna):
            raise ConfigurationError("Invalid Optuna parameter definitions")

        patience_val = self.cfg.optuna.get("patience", 5)
        self.patience = (
            int(patience_val) if isinstance(patience_val, (int, float)) else 5
        )

    def _initialize_static_data(self) -> None:
        logger.info("Loading static data from %s", self.csv_path)
        full_df = load_and_clean_data(
            self.csv_path, self.cfg, enforce_min_size=MIN_DATASET_SIZE
        )
        train_df, val_df = stratified_split(full_df, 0.2, seed=self.seed)

        self.train_dataset, self.val_dataset = build_two_tower_datasets(
            train_df, val_df, self.cfg, self.dims, preload_ram=False
        )
        self.drug_dim, self.geno_dim = infer_dimensions(self.train_dataset, self.cfg)
        self.axes = infer_axis_specs(
            self.train_dataset.target_encoder.encoders,
            self.train_dataset.targets,
            set(get_settings().multi_label_set),
            get_axes_config(),
        )
        logger.info(
            "Inferred dimensions: drug=%d, geno=%d", self.drug_dim, self.geno_dim
        )
        logger.info("Inferred axes: %s", list(self.axes.keys()))

        del full_df, train_df, val_df
        gc.collect()

    def _suggest_params(self, trial: optuna.Trial) -> dict[str, Any]:
        suggestions: dict[str, Any] = {}
        for param_name, spec in self.cfg.optuna.items():
            if not hasattr(spec, "kind"):
                continue
            if spec.kind == "categorical":
                suggestions[param_name] = trial.suggest_categorical(
                    param_name, spec.choices
                )
            elif spec.kind == "int":
                suggestions[param_name] = trial.suggest_int(
                    param_name, spec.low, spec.high
                )
            elif spec.kind == "float":
                suggestions[param_name] = trial.suggest_float(
                    param_name, spec.low, spec.high
                )
            elif spec.kind == "log":
                suggestions[param_name] = trial.suggest_float(
                    param_name, spec.low, spec.high, log=True
                )
        return suggestions

    def _get_epochs(self) -> int:
        if self.epochs_override is not None:
            return self.epochs_override
        epochs_val = self.cfg.optuna.get("epochs", 50)
        return int(epochs_val) if isinstance(epochs_val, (int, float)) else 50

    def _build_pipeline(self, params: dict[str, Any]) -> OptunaTrialTrainer:
        model = build_gnn_model(
            model_name=self.model_name,
            dims=self.dims,
            drug_dim=self.drug_dim,
            geno_dim=self.geno_dim,
            axes=self.axes,
            params=params,
            device=self.device,
        )

        multitask_loss = MultiTaskLoss(self.axes).to(self.device)
        compose_loss = CompositionalLabelLoss()
        optimizer = OptimizerFactory.create(
            model=model, params=params, uncertainty_module=multitask_loss
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        return OptunaTrialTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=self.device,
            target_cols=self.cfg.targets,
            multi_label_cols=get_settings().multi_label_set,
            params=params,
            multitask_loss=multitask_loss,
            compose_loss=compose_loss,
            compose_weight=params.get("compose_loss_weight", 0.5),
        )

    def objective(self, trial: optuna.Trial) -> float:
        params = self._suggest_params(trial)
        batch_size = min(int(params.get("batch_size", 64)), self.max_batch_size)
        epochs = self._get_epochs()

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collater,
            num_workers=0,
            pin_memory=True,
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collater,
            num_workers=0,
            pin_memory=True,
        )

        trainer = self._build_pipeline(params)
        return trainer.fit(
            train_loader,
            val_loader,
            epochs=epochs,
            patience=self.patience,
            trial=trial,
        )

    def tune(self, n_trials: int, n_jobs: int | None = 1) -> optuna.Study:
        sampler = (
            TPESampler(seed=self.seed, multivariate=True)
            if self.sampler_type == "TPE"
            else RandomSampler(seed=self.seed)
        )
        pruner = (
            PatientPruner(HyperbandPruner(), patience=5)
            if self.pruner_type == "Hyperband"
            else MedianPruner()
        )

        storage_url = (
            f"sqlite:///{self.reports_dir.parent}/database/{self.study_name}.db"
        )
        Path(storage_url.replace("sqlite:///", "")).parent.mkdir(
            parents=True, exist_ok=True
        )

        study = optuna.create_study(
            study_name=self.study_name,
            storage=storage_url,
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )

        logger.info("Starting Optuna study: %s", self.study_name)
        if n_jobs is None:
            n_jobs = 1
        if n_jobs > 1:
            study.optimize(
                self.objective,
                n_trials=n_trials,
                n_jobs=n_jobs,
                gc_after_trial=True,
            )
        else:
            with tqdm(total=n_trials, desc="Trials", colour="blue") as pbar:

                def callback(study, trial):
                    pbar.update(1)
                    try:
                        best = study.best_trial
                        pbar.set_postfix(
                            {"Best": f"{best.value:.4f}", "Trial": trial.number}
                        )
                    except ValueError:
                        pbar.set_postfix({"Status": trial.state.name})

                study.optimize(
                    self.objective,
                    n_trials=n_trials,
                    n_jobs=1,
                    callbacks=[callback],
                    gc_after_trial=True,
                )

        self._save_results(study)
        return study

    def _save_results(self, study: optuna.Study) -> None:
        completed_trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if not completed_trials:
            logger.warning("No trials completed successfully. Skipping report.")
            return

        is_minimize = study.direction == optuna.study.StudyDirection.MINIMIZE
        best_trial = study.best_trial
        completed_trials.sort(key=lambda t: t.value, reverse=not is_minimize)  # type: ignore
        top_5 = completed_trials[:5]

        report: dict[str, Any] = {
            "model": self.model_name,
            "targets": self.cfg.targets,
            "direction": study.direction.name,
            "best_trial": {
                "number": best_trial.number,
                "value": best_trial.value,
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
                "n_pruned": len(
                    [
                        t
                        for t in study.trials
                        if t.state == optuna.trial.TrialState.PRUNED
                    ]
                ),
            },
            "top_5_trials": [
                {"trial": t.number, "value": t.value, "params": t.params} for t in top_5
            ],
            "configuration": {
                "sampler": self.sampler_type,
                "pruner": self.pruner_type,
                "max_batch_size": self.max_batch_size,
                "random_seed": self.seed,
            },
            "datetime": self.timestamp,
        }

        json_path = (
            self.reports_dir
            / "reports"
            / f"optuna_{self.study_name.replace('OPT_', '')}.json"
        )
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)
        ConsoleIO.print_success(f"Report saved: {json_path}")

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from optuna.visualization.matplotlib import (
                plot_optimization_history,
                plot_param_importances,
            )

            plt.figure(figsize=(10, 6))
            plot_optimization_history(study)
            plt.title(f"Optimization History - {self.model_name}")
            plt.tight_layout()
            history_path = self.figures_dir / f"{self.study_name}_history.png"
            plt.savefig(history_path, dpi=300)
            plt.close()
            logger.info("History plot saved to %s", history_path)

            if len(completed_trials) > 1:
                plt.figure(figsize=(10, 6))
                plot_param_importances(study)
                plt.title(f"Parameter Importance - {self.model_name}")
                plt.tight_layout()
                importance_path = self.figures_dir / f"{self.study_name}_importance.png"
                plt.savefig(importance_path, dpi=150)
                plt.close()
                logger.info("Importance plot saved to %s", importance_path)

            ConsoleIO.print_success("Optimization plots generated successfully")
            ConsoleIO.print_info(f"Plots saved in: {self.figures_dir}")

        except ImportError:
            logger.warning(
                "Matplotlib not available; skipping Optuna visualisation plots."
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
    if mp.get_start_method(allow_none=True) != "spawn":
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

    tuner = PGenTuner(
        model_name=model_name,
        csv_path=csv_path,
        sampler_type=sampler,
        pruner_type=pruner,
        epochs_override=epochs,
    )
    return tuner.tune(n_trials=n_trials, n_jobs=N_JOBS)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:  # noqa
        print("Usage: python tuner.py <model_name> [csv_path] [n_trials]")
        sys.exit(1)

    run_optuna_study(
        model_name=sys.argv[1],
        csv_path=sys.argv[2] if len(sys.argv) > 2 else "train_data/train_data.tsv",  # noqa
        n_trials=int(sys.argv[3]) if len(sys.argv) > 3 else 50,  # noqa
    )
