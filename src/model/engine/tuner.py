# Pharmagen - Optuna Hyperparameter Optimization
# Copyright (C) 2025 Adrim Hamed Outmani

"""Optuna search over the ``[Model.optuna]`` space of ``models.toml``.

The data is prepared once (:func:`prepare_training_data`); every trial builds
a fresh model + :class:`Trainer` from the suggested hyperparameters layered
over the model's fixed ``params`` and minimizes the validation loss.
"""

from __future__ import annotations

import datetime
import json
import logging
from typing import TYPE_CHECKING, Any, Literal

import optuna
from optuna.pruners import HyperbandPruner, MedianPruner, NopPruner, PatientPruner
from optuna.samplers import CmaEsSampler, RandomSampler, TPESampler
from torch.utils.data import DataLoader

from src.config import get_model_config, get_settings
from src.config.models import CategoricalSpec, FloatSpec, IntSpec, LogSpec
from src.core import ConfigurationError, ConfigValidator
from src.data.collator import DoubleTowerCollater
from src.model.engine.base import (
    build_gnn_model,
    build_trainer,
    model_switches,
    prepare_training_data,
    resolve_device,
)

if TYPE_CHECKING:
    from pathlib import Path

    from src.evaluation.runs import FoldRef
    from src.model.engine.base import DataSource

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

MIN_DATASET_SIZE = 1000
DEFAULT_MAX_BATCH_SIZE = 128
DEFAULT_PATIENCE = 5
DEFAULT_EPOCHS = 50

SamplerName = Literal["tpe", "random", "cmaes"]
PrunerName = Literal["hyperband", "median", "none"]


def _make_sampler(name: SamplerName, seed: int) -> optuna.samplers.BaseSampler:
    if name == "random":
        return RandomSampler(seed=seed)
    if name == "cmaes":
        return CmaEsSampler(seed=seed)
    return TPESampler(seed=seed, multivariate=True)


def _make_pruner(name: PrunerName) -> optuna.pruners.BasePruner:
    if name == "median":
        return MedianPruner()
    if name == "none":
        return NopPruner()
    return PatientPruner(HyperbandPruner(), patience=5)


class PGenTuner:
    """Optuna study for one model configuration."""

    def __init__(
        self,
        model_name: str,
        source: DataSource,
        *,
        seed: int | None = None,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
        epochs: int | None = None,
        split: FoldRef | None = None,
    ) -> None:
        self.model_name = model_name
        self.seed = seed if seed is not None else get_settings().seed
        self.max_batch_size = max_batch_size
        self.device = resolve_device()

        self.cfg = get_model_config(model_name)
        if not self.cfg.optuna:
            raise ConfigurationError(
                f"Model '{model_name}' missing 'optuna' search space"
            )
        if not ConfigValidator.validate_optuna_params(self.cfg.optuna):
            raise ConfigurationError("Invalid Optuna parameter definitions")
        self.patience = int(self.cfg.optuna.get("patience", DEFAULT_PATIENCE))
        self.epochs = epochs or int(self.cfg.optuna.get("epochs", DEFAULT_EPOCHS))
        self.switches = model_switches(self.cfg)

        self.data = prepare_training_data(
            self.cfg,
            source,
            seed=self.seed,
            preload_ram=False,
            min_size=MIN_DATASET_SIZE,
            split=split,
        )
        self.collater = DoubleTowerCollater()

    def _suggest_params(self, trial: optuna.Trial) -> dict[str, Any]:
        suggestions: dict[str, Any] = {}
        for name, spec in self.cfg.optuna.items():
            if isinstance(spec, CategoricalSpec):
                suggestions[name] = trial.suggest_categorical(name, spec.choices)
            elif isinstance(spec, IntSpec):
                suggestions[name] = trial.suggest_int(name, spec.low, spec.high)
            elif isinstance(spec, LogSpec):
                suggestions[name] = trial.suggest_float(
                    name, spec.low, spec.high, log=True
                )
            elif isinstance(spec, FloatSpec):
                suggestions[name] = trial.suggest_float(name, spec.low, spec.high)
        return suggestions

    def _loader(self, dataset: Any, batch_size: int, *, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collater,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
        )

    def objective(self, trial: optuna.Trial) -> float:
        params = self.cfg.training_params(self._suggest_params(trial))
        batch_size = min(params.batch_size, self.max_batch_size)
        model = build_gnn_model(
            dims=self.data.dims,
            drug_dim=self.data.drug_dim,
            geno_dim=self.data.geno_dim,
            axes=self.data.axes,
            params=params,
            device=self.device,
            switches=self.switches,
            gene_vocab=self.data.gene_vocab,
        )
        trainer = build_trainer(model, self.cfg, self.data.axes, params, self.device)
        return trainer.fit(
            self._loader(self.data.train, batch_size, shuffle=True),
            self._loader(self.data.val, batch_size, shuffle=False),
            epochs=self.epochs,
            patience=self.patience,
            trial=trial,
        )

    def tune(
        self,
        n_trials: int,
        *,
        sampler: SamplerName = "tpe",
        pruner: PrunerName = "hyperband",
        storage: str | None = None,
        study_name: str | None = None,
        n_jobs: int = 1,
    ) -> optuna.Study:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        study_name = study_name or f"OPT_{self.model_name}_{stamp}"
        if storage is None:
            db_dir = get_settings().paths.reports / "optuna" / "database"
            db_dir.mkdir(parents=True, exist_ok=True)
            storage = f"sqlite:///{db_dir / f'{study_name}.db'}"

        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="minimize",
            sampler=_make_sampler(sampler, self.seed),
            pruner=_make_pruner(pruner),
            load_if_exists=True,
        )
        logger.info("Starting Optuna study %s (%d trials)", study_name, n_trials)
        study.optimize(
            self.objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            gc_after_trial=True,
            show_progress_bar=n_jobs == 1,
        )
        self._save_results(study)
        return study

    def _save_results(self, study: optuna.Study) -> Path | None:
        states = optuna.trial.TrialState
        completed = [t for t in study.trials if t.state == states.COMPLETE]
        if not completed:
            logger.warning("No trials completed successfully; skipping report")
            return None

        completed.sort(key=lambda t: t.value)  # type: ignore[arg-type, return-value]
        best = study.best_trial
        report = {
            "model": self.model_name,
            "study": study.study_name,
            "targets": self.cfg.targets,
            "best_trial": {
                "number": best.number,
                "value": best.value,
                "params": best.params,
            },
            "statistics": {
                "n_trials": len(study.trials),
                "n_completed": len(completed),
                "n_pruned": sum(t.state == states.PRUNED for t in study.trials),
            },
            "top_5_trials": [
                {"trial": t.number, "value": t.value, "params": t.params}
                for t in completed[:5]
            ],
            "seed": self.seed,
        }
        reports_dir = get_settings().paths.reports / "optuna"
        reports_dir.mkdir(parents=True, exist_ok=True)
        json_path = reports_dir / f"{study.study_name}.json"
        json_path.write_text(json.dumps(report, indent=2))
        logger.info("Optuna report saved to %s", json_path)
        self._save_plots(study, reports_dir, plot_importance=len(completed) > 1)
        return json_path

    def _save_plots(
        self, study: optuna.Study, out_dir: Path, *, plot_importance: bool
    ) -> None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from optuna.visualization.matplotlib import (
                plot_optimization_history,
                plot_param_importances,
            )
        except ImportError:
            logger.warning("matplotlib not installed; skipping Optuna plots")
            return

        plots = [("history", plot_optimization_history)]
        if plot_importance:
            plots.append(("importance", plot_param_importances))
        for name, plot in plots:
            try:
                plot(study)
                path = out_dir / f"{study.study_name}_{name}.png"
                plt.tight_layout()
                plt.savefig(path, dpi=150)
                logger.info("Optuna %s plot saved to %s", name, path)
            except Exception as e:  # noqa: BLE001 — plotting must not fail a study
                logger.warning("Optuna %s plot failed: %s", name, e)
            finally:
                plt.close("all")


def run_optuna_study(
    model_name: str,
    source: DataSource,
    n_trials: int = 50,
    *,
    epochs: int | None = None,
    sampler: SamplerName = "tpe",
    pruner: PrunerName = "hyperband",
    storage: str | None = None,
    study_name: str | None = None,
    n_jobs: int = 1,
    split: FoldRef | None = None,
) -> optuna.Study:
    """Prepare the data once and run ``n_trials`` Optuna trials.

    With ``split`` the search trains and validates on that scenario fold.
    """
    tuner = PGenTuner(model_name, source, epochs=epochs, split=split)
    return tuner.tune(
        n_trials,
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        study_name=study_name,
        n_jobs=n_jobs,
    )


__all__ = ["PGenTuner", "run_optuna_study"]
