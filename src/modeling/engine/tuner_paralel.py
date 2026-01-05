# tuner.py
# Pharmagen - Optuna Hyperparameter Optimization
#
# Implements a comprehensive hyperparameter optimization pipeline using Optuna.
# Supports multi-objective optimization and clean architecture.
# Copyright (C) 2025 Adrim Hamed Outmani

import datetime
import json
import logging
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import optuna.logging
import optuna.study as opt_study
import torch
from optuna.pruners import HyperbandPruner, PatientPruner
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Project Imports
from src.config.manager import DIRS, get_model_config
from src.data.loaders import DoubleTowerCollater, DoubleTowerDataset
from src.modeling.architectures.layers import create_gnn_model
from src.modeling.engine.trainer import PGenTrainer, TrainerConfig
from src.utils.io import DataLoaderUtils
from src.utils.module_builder import LossFactory, OptimizerFactory

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

class PGenTuner:
    """
    Orchestrator Class for Hyperparameter Optimization.

    Encapsula la complejidad de la preparación de datos, la definición del espacio de búsqueda,
    la ejecución del ciclo de vida de Optuna y la generación de reportes.
    """

    def __init__(
        self, model_name: str, csv_path: str | Path, random_seed: int = 711
    ):
        self.model_name = model_name
        self.csv_path = Path(csv_path)
        self.timestamp = datetime.datetime.now().strftime("%d_%m__%H_%M")
        self.study_name = f"OPT_{self.model_name}_{self.timestamp}"
        self.seed = random_seed

        # 1. Configuración Global y Dispositivo
        self.cfg = get_model_config(model_name)
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.patience = self.cfg["params_optuna"].get("patience", 5)

        # Directorios de salida (Asegurar existencia)
        self.reports_dir = DIRS.get("reports", Path("./reports")) / "optuna_reports"
        self.figures_dir = DIRS.get("reports", Path("./reports")) / "figures"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        # 2. Carga y Preparación de Datos (Solo una vez)
        logger.info(f"Loading data for tuning from {self.csv_path}...")
        self.full_df = DataLoaderUtils.load_dataframe(
            self.csv_path,
            cols=self.cfg["cols"],
            stratify_col=self.cfg.get("stratify_col", None),
        )

        # stratify = self.full_df["_stratify"] if "_stratify" in self.full_df.columns else None

        self.train_df, self.val_df = train_test_split(
            self.full_df, test_size=0.2, stratify=None, random_state=self.seed
        )
        logger.info(
            f"Tuning Data Ready: {len(self.train_df)} train, {len(self.val_df)} val"
        )

    def _calculate_parallel_jobs(self, estimated_vram_per_trial_gb: float = 2.3) -> int:
        """
        Calcula n_jobs basado en la RTX 4070 Ti Super (16GB) y CPU disponible.
        """
        if not torch.cuda.is_available():
            return 1 # Fallback a CPU serial

        # VRAM Total en GB
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        # Reservamos 2GB para sistema/overhead
        usable_vram = total_vram - 2.0

        gpu_jobs = int(usable_vram // estimated_vram_per_trial_gb)
        cpu_cores = os.cpu_count() or 1

        # Limitamos por CPU también (evitar thrashing)
        # Dejamos 2 cores libres para el sistema
        max_cpu_jobs = max(1, cpu_cores - 2)

        n_jobs = max(1, min(gpu_jobs, max_cpu_jobs))

        logger.info(f"[Auto-Scale] Detected {total_vram:.1f}GB VRAM. Setting n_jobs={n_jobs} (Est. Trial: {estimated_vram_per_trial_gb}GB)")
        return n_jobs

    def _suggest_params(self, trial: optuna.Trial) -> dict[str, Any]:
        """
        Parser dinámico del espacio de búsqueda definido en la configuración (JSON/TOML).
        Abstrae los métodos de sugerencia de Optuna.
        """
        suggestions = {}
        optuna_conf = self.cfg.get("params_optuna", {})

        for param_name, args in optuna_conf.items():
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
            except Exception as e:
                logger.warning(f"Error parsing param {param_name}: {e}. Using default.")
        return suggestions

    def objective(self, trial: optuna.Trial) -> float:
        """
        Función objetivo para Optuna.
        Instancia Modelo, Dataset y Trainer frescos para cada trial.
        """
        # 1. Configuración de Hiperparámetros
        params = self._suggest_params(trial)
        #params.update(self._suggest_params(trial))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        batch_size = int(params.get("batch_size", 64))
        epochs = int(self.cfg.get("params_optuna", {}).get("epochs", 50))

        # 2. Construcción de Datasets (Reutilizando encoders para consistencia)
        # Nota: preload_ram=False ahorra memoria en la GPU si los grafos son muchos
        train_dataset = DoubleTowerDataset(
            df=self.train_df,
            drug_col=self.cfg["features"][0],
            haplo_col="haplo_key",
            target_cols=self.cfg["targets"],
            multilabel_cols=self.cfg.get("multi_label_cols", []),
            preload_ram=False,
        )

        val_dataset = DoubleTowerDataset(
            df=self.val_df,
            drug_col=self.cfg["features"][0],
            haplo_col="haplo_key",
            target_cols=self.cfg["targets"],
            multilabel_cols=self.cfg.get("multi_label_cols", []),
            encoders=train_dataset.encoders,  # CRÍTICO: Compartir estado de encoders
            preload_ram=False,
        )

        # 3. Inferencia de Dimensiones
        try:
            sample = train_dataset[0]
            drug_dim = sample["drug_data"].x.shape[1]  # type: ignore
            haplo_dim = sample["haplo_data"].x.shape[1]  # type: ignore
        except Exception as e:
            logger.error(f"Dimension check failed: {e}")
            raise optuna.exceptions.TrialPruned()

        target_dims = {
            col: len(train_dataset.encoders[col].classes_)
            for col in self.cfg["targets"]
            if col in train_dataset.encoders
        }
        # Fallback para dimensiones target si algo falla
        for t in self.cfg["targets"]:
            if t not in target_dims:
                target_dims[t] = 1

        # 4. DataLoaders con Collater Especializado
        collater = DoubleTowerCollater()
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collater,
            num_workers=0,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collater,
            num_workers=0,
            pin_memory=True,
        )

        # 5. Instanciación del Modelo (Two-Tower GATv2)
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

        # 5. Loss & Uncertainty Setup (IMPORTANTE)
        uncertainty_net = LossFactory.create_uncertainty_wrapper(
            tasks=self.cfg["targets"], device=device
        )
        # 6. Optimizer & Scheduler (Usando la Factoría)
        optimizer = OptimizerFactory.create(
            model=model,
            params=params,  # Aquí se inyectan learning_rate y weight_decay de Optuna
            uncertainty_module=uncertainty_net,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        # 7. Trainer Setup (Inyectando uncertainty_module)
        trainer_config = TrainerConfig(
            device=device,
            target_cols=self.cfg["targets"],
            multi_label_cols=set(self.cfg.get("multi_label_cols", [])),
            params=params,
            from_optuna=True,
        )

        trainer = PGenTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=trainer_config,
            uncertainty_module=uncertainty_net,  # <--- FIX: Ahora el tuner usa incertidumbre
        )
        try:
            return trainer.fit(
                train_loader,
                val_loader,
                epochs=epochs,
                patience=self.patience,
                trial=trial,
            )
        except optuna.exceptions.TrialPruned:
            raise
        except Exception as e:
            logger.error(f"Trial failed: {e}")
            raise optuna.exceptions.TrialPruned()

    def run_tuning(self, n_trials: int = 50, n_jobs: int | None = None) -> optuna.Study:
        """
        Ejecuta el estudio completo.
        """
        if n_jobs is None:
            n_jobs = self._calculate_parallel_jobs()

        logger.info(f"Starting Optuna Study: {self.study_name} w/ n_jobs={n_jobs}")

        # Sampler TPE (Tree-structured Parzen Estimator) es ideal para hiperparámetros
        sampler = TPESampler(seed=self.seed, multivariate=True)
        no_patience_pruner = HyperbandPruner(min_resource=5, max_resource=50, reduction_factor=3)
        pruner = PatientPruner(no_patience_pruner, patience=5)

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
            logger.info("Running in parallel mode. TQDM bar disabled to prevent console clutter.")

            study.optimize(self.objective, n_trials=n_trials, n_jobs=n_jobs, gc_after_trial=True)
        else:
            # Barra de progreso externa
            with tqdm(total=n_trials, desc="Optuna Trials", colour="blue") as pbar:
                def callback(study, trial):
                    pbar.update(1)
                    if study.best_trial:
                        best_trial = study.best_trial
                        best_trial_params = [f"{k}: {v}" for k, v in best_trial.params.items()]
                        postfix = {
                            "Trial": best_trial.number,
                            "Best Loss": f"{best_trial.value:.4f}",
                            "Params": " ".join(best_trial_params),
                        }
                        pbar.set_postfix(postfix)
                study.optimize(
                    self.objective,
                    n_trials=n_trials,
                    callbacks=[callback],
                    gc_after_trial=True,
                )

        self._save_results(study)
        return study

    def _save_results(self, study: optuna.Study):
        """Genera reportes JSON y gráficas."""
        # 1. JSON Report
        best_trial = study.best_trial
        top_5 = study.trials[1:6]
        report = {
            "model": self.model_name,
            "best_value": best_trial.value,
            "best_params": best_trial.params,
            "n_trials": len(study.trials),
            "datetime": self.timestamp,
            "top_5_trials": [{"value": t.value, "params": t.params} for t in top_5],
        }

        json_path = self.reports_dir / f"report_{self.study_name}.json"
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Metrics saved to {json_path}")

        # 2. Visualizaciones (Si matplotlib está disponible)
        try:
            from optuna.visualization.matplotlib import (
                plot_optimization_history,
                plot_param_importances,
            )

            # History
            plt.figure(figsize=(10, 6))
            plot_optimization_history(study)
            plt.title(f"Optimization History - {self.model_name}")
            plt.tight_layout()
            plt.savefig(self.figures_dir / f"{self.study_name}_history.png")
            plt.close()

            # Importance (Solo si hay suficientes trials completos)
            if (
                len(
                    [
                        t
                        for t in study.trials
                        if t.state == optuna.trial.TrialState.COMPLETE
                    ]
                )
                > 1
            ):
                plt.figure(figsize=(10, 6))
                plot_param_importances(study)
                plt.title(f"Param Importance - {self.model_name}")
                plt.tight_layout()
                plt.savefig(self.figures_dir / f"{self.study_name}_importance.png")
                plt.close()

            logger.info("Optimization plots generated.")

        except ImportError:
            logger.warning(
                "Matplotlib not found inside optuna.visualization, skipping plots."
            )
        except Exception as e:
            logger.warning(f"Plot generation failed: {e}")


# Entry Point Simplificado
def run_optuna_study(model_name: str, csv_path: str | Path, n_trials: int = 50):
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    print(Path(csv_path).resolve())
    print(f"Running Optuna Study for {model_name} on data {csv_path} with {n_trials} trials.")
    print("-" * 50)
    print(os.getcwd())

    tuner = PGenTuner(model_name=model_name, csv_path=csv_path)
    study = tuner.run_tuning(n_trials=n_trials, n_jobs=6)

    print(f"\n[Optuna] Best Params: {study.best_params}")

if __name__ == "__main__":
    run_optuna_study("TwoTowerGAT", "train_data/train_data.csv", 100)

    print("Finalizado")
