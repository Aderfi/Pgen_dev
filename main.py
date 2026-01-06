# Pharmagen - Pharmacogenetic Prediction and Therapeutic Efficacy
# Copyright (C) 2025 Adrim Hamed Outmani
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
#!/usr/bin/env python3
# coding=utf-8
"""
Pharmagen - Punto de Entrada Principal (CLI & Orquestador).

Este script actúa como la interfaz principal de ejecución para el software Pharmagen.
Su responsabilidad es inicializar el entorno, configurar el sistema de logging global
y enrutar la solicitud del usuario hacia el módulo correspondiente (Entrenamiento,
Predicción o Interfaz Interactiva).

Uso:
    El script puede ejecutarse en dos modalidades:
    1. Interactivo (Por defecto): Lanza un menú visual.
    2. Headless (CLI): Ejecuta tareas específicas mediante argumentos.

Ejemplos:
    # 1. Iniciar menú interactivo
    $ python main.py

    # 2. Entrenar un modelo específico automáticamente
    $ python main.py --mode train --model Phenotype_Effect_Outcome --input data/train.tsv

    # 3. Realizar predicciones sobre un archivo nuevo
    $ python main.py --mode predict --model Phenotype_Effect_Outcome --input data/pacientes.csv

Author:
    Adrim Hamed Outmani (@Aderfi)

Copyright:
    (C) 2025 Adrim Hamed Outmani. Licensed under GNU GPLv3.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import cast

# --- Imports del Proyecto ---
from src.config.manager import DIRS
from src.interface.cli import main_menu_loop
from src.utils.logger import setup_logging

# from src.utils.system import check_environment_and_setup

# --- Setup de Rutas ---
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

# Constantes
DATE_STAMP = datetime.now().strftime("%Y-%m-%d")
LOGS_DIR = DIRS["logs"]
# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def arguments_parser() -> argparse.ArgumentParser:
    parser = argparse. ArgumentParser(description="Pharmagen CLI Manager")
    parser.add_argument(
        "--mode", "-m",
        choices=["train", "predict", "menu"],  # Fixed: replaced "optuna" with "predict"
        default="menu",
        help="Modo de ejecución",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="TwoTowerGAT",
        help="Nombre del modelo (para automatización)",
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("train_data/train_data.tsv"),
        help="Ruta al archivo de entrada (CSV/TSV)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        const=logging.INFO,
        default=False,
        help="Habilitar salida detallada",
    )

    parser.add_argument(
        '--debug',
        action='store_const',
        const=logging.DEBUG,
        default=logging.WARNING,)

    parser.add_argument(
        "--optuna", "-opt",
        action="store_true",
        help="Usar Optuna para optimización de hiperparámetros (solo en modo train)",
    )

    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=100,
        metavar="N",
        help="Número de trials para Optuna (default: %(default)s)",
    )

    parser.add_argument(
        "--optuna-epochs",
        type=int,
        default=50,
        metavar="N",
        help="Número de epochs por trial en Optuna (default: %(default)s)",
    )


    return parser

def main(args: argparse.Namespace | None = None):
    logger = logging.getLogger("Pharmagen")
    logger.setLevel(logging.DEBUG)

    if args is None:
        args = arguments_parser().parse_args()

    args = cast(argparse.Namespace, args)
    try:
        # Interactive Menu (Default)
        if args.mode == "menu":
            main_menu_loop()

        # Training Mode (Headless/Automated)
        elif args.mode == "train":
            if not args.model:
                args.model = args.model.default
                print("❌ Error: --model es obligatorio en modo 'train'")
                sys.exit(1)

            logger.info(f"Iniciando entrenamiento headless: {args.model}")
            if args. optuna:
                from src. modeling.engine.tuner import run_optuna_study

                run_optuna_study(args. model, args.input)
            else:
                from src.pipeline import train_pipeline

                train_pipeline(model_name=args. model, csv_path=args.input)

        # Modo Predicción (Headless/Automatizado)
        elif args.mode == "predict":
            import pandas as pd

            from src.modeling.engine.predictor import PGenPredictor

            if not args.model:
                print("❌ Error: --model es obligatorio en modo 'predict'")
                sys.exit(1)

            logger.info(f"Iniciando predicción headless: {args. model}")
            predictor = PGenPredictor(args.model)
            results = predictor. predict_file(args.input)

            # Guardado automático
            out_name = f"{Path(args.input).stem}_preds_{DATE_STAMP}. csv"
            results_df = pd.DataFrame(data=results)
            results_df.to_csv(out_name, index=False)
            print(f"Predicciones guardadas en:  {out_name}")

    except KeyboardInterrupt:
        print("\nOperación cancelada por el usuario.")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Error no controlado en Main:  {e}", exc_info=True)
        print(f"\n❌ Error crítico del sistema: {e}")
        print(f"Consulte el log para más detalles: {LOGS_DIR}")
        sys.exit(1)


if __name__ == "__main__":
    # check_environment_and_setup()
    setup_logging()
    main()
