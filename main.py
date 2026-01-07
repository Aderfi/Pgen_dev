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
from src.interface.ui import ConsoleIO, Spinner
from src.utils.logger import setup_logging

# from src.utils.system import check_environment_and_setup

# --- Setup de Rutas ---
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

# Constantes
DATE_STAMP = datetime.now().strftime("%Y-%m-%d")
LOGS_DIR = DIRS["logs"]

# Logger
logger = logging.getLogger("Pharmagen")

# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def arguments_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pharmagen CLI Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=
        """
            Examples:
            # Interactive menu
            python main.py

            # Standard training
            python main.py --mode train --model TwoTowerGAT --input data/train.tsv

            # Optuna optimization
            python main.py --mode train --optuna --optuna-trials 50 --optuna-epochs 30

            # Prediction
            python main.py --mode predict --model TwoTowerGAT --input data/test.csv
            """,
        )

    parser.add_argument(
        '--mode', '-m',
        type=str,
        choices=["train", "predict", "menu"],
        default="menu",
        help="Execution mode:  train, predict, or interactive menu (default: menu)",
    )

    parser.add_argument(
        '--model',
        type=str,
        default="TwoTowerGAT",
        help="Model name for training/prediction (default: TwoTowerGAT)",
    )

    parser.add_argument(
        '--input',
        type=Path,
        default=Path("train_data/train_data.tsv"),
        help="Input data file path (CSV/TSV) (default: train_data/train_data.tsv)",
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        metavar="N",
        help="Number of training epochs (default: 100)",
    )

    parser.add_argument(
        '--verbose', '-v',
        action="store_true",
        help="Enable verbose output (INFO level logging)",
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help="Enable debug output (DEBUG level logging)",
    )

    parser.add_argument(
        "--optuna", "-opt",
        action="store_true",
        help="Use Optuna for hyperparameter optimization (only in train mode)",
    )

    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=100,
        metavar="N",
        help="Number of Optuna trials (default: 100)",
    )

    parser.add_argument(
        "--optuna-epochs",
        type=int,
        default=50,
        metavar="N",
        help="Number of epochs per Optuna trial (default: 50)",
    )

    return parser


def _run_headless_training(args: argparse.Namespace):
    """Execute training in headless mode."""
    # Validate input file exists
    if not args.input.exists():
        ConsoleIO.print_error(f"Input file not found: {args.input}")
        sys.exit(1)

    # Standard Training
    if not args.optuna:
        from src.pipeline import train_pipeline

        logger.info(f"Starting standard training:  {args.model}")
        ConsoleIO.print_header("Standard Training")
        ConsoleIO.print_info(f"Model: {args.model}")
        ConsoleIO.print_info(f"Data:  {args.input}")
        ConsoleIO.print_info(f"Epochs: {args.epochs}")

        with Spinner(f"Training {args.model}.. .", style="braille"):
            train_pipeline(
                model_name=args.model,
                csv_path=str(args.input),
                epochs=args.epochs
            )
        
        ConsoleIO.print_success("Training completed successfully!")

    # Optuna Optimization
    else:
        from src.modeling.engine.tuner import run_optuna_study

        logger.info(f"Starting Optuna optimization: {args.model}")
        ConsoleIO.print_header("Optuna Hyperparameter Optimization")
        ConsoleIO.print_info(f"Model: {args.model}")
        ConsoleIO.print_info(f"Data: {args.input}")
        ConsoleIO.print_info(f"Trials: {args.optuna_trials}")
        ConsoleIO.print_info(f"Epochs per trial: {args.optuna_epochs}")

        run_optuna_study(
            model_name=args.model,
            csv_path=str(args.input),
            n_trials=args.optuna_trials,
            epochs=args.optuna_epochs
        )
        
        ConsoleIO.print_success("Optuna optimization completed!")


def _run_headless_prediction(args: argparse.Namespace):
    """Execute prediction in headless mode."""
    import pandas as pd
    from src.modeling.engine.predictor import PGenPredictor

    # Validate input file exists
    if not args.input.exists():
        ConsoleIO.print_error(f"Input file not found:  {args.input}")
        sys.exit(1)

    logger.info(f"Starting headless prediction: {args.model}")
    ConsoleIO.print_header("Prediction Mode")
    ConsoleIO.print_info(f"Model: {args.model}")
    ConsoleIO.print_info(f"Input: {args.input}")

    try:
        # Load model
        with Spinner("Loading model...", style="braille"):
            predictor = PGenPredictor(args.model)
        
        ConsoleIO.print_success("Model loaded successfully")

        # Run predictions
        with Spinner(f"Processing {args.input.name}...", style="braille"):
            results = predictor.predict_file(args.input)

        if not results:
            ConsoleIO.print_warning("No predictions generated")
            return

        # Save results
        out_name = f"{args.input.stem}_predictions_{DATE_STAMP}. csv"
        out_path = args.input.parent / out_name
        
        results_df = pd.DataFrame(data=results)
        results_df.to_csv(out_path, index=False)
        
        ConsoleIO.print_success(f"Predictions saved to: {out_path}")
        ConsoleIO.print_info(f"Total predictions: {len(results)}")

    except FileNotFoundError as e:
        logger.error(f"Model not found: {e}")
        ConsoleIO.print_error(f"Model '{args.model}' not found")
        ConsoleIO.print_info("Tip: Train the model first using --mode train")
        sys.exit(1)


def main(args: argparse.Namespace | None = None):
    """
    Main entry point for Pharmagen CLI.

    Args:
        args: Parsed command line arguments (optional, will parse if None)
    """
    if args is None:
        args = arguments_parser().parse_args()

    args = cast(argparse.Namespace, args)

    # Configure logging level based on flags
    log_level = logging.WARNING
    if args.debug:
        log_level = logging.DEBUG
    elif args.verbose:
        log_level = logging.INFO

    # Configure logging once
    setup_logging(name="Pharmagen", level=log_level)
    logger.setLevel(log_level)

    try:
        # =====================================================================
        # MODE:  Interactive Menu (Default)
        # =====================================================================
        if args. mode == "menu": # Interactive Menu (Default)
            main_menu_loop()

        elif args.mode == "train": # Training (Headless/Automated)
            _run_headless_training(args)

        elif args.mode == "predict": # Prediction (Headless/Automated)
            _run_headless_prediction(args)

    except KeyboardInterrupt:
        ConsoleIO.print_warning("\nOperation cancelled by user.")
        logger.info("User interrupted execution")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Unhandled error in main:  {e}", exc_info=True)
        ConsoleIO.print_error(f"Critical system error: {e}")
        ConsoleIO.print_info(f"Check logs for details: {LOGS_DIR}")
        sys.exit(1)

if __name__ == "__main__":
    main()
