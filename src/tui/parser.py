# src/tui/parser.py
# Pharmagen - Headless CLI argument parser

"""Argument parser for the headless Pharmagen CLI.

The parser is isolated from dispatch so it can be unit-tested and reused
without importing training/prediction dependencies.
"""

import argparse
from pathlib import Path

_DEFAULT_INPUT = Path("train_data/train_data.tsv")

_EPILOG = """\
Examples:
  # Standard training
  python main.py train -M TwoTowerGAT -i data/train.tsv

  # Optuna optimization
  python main.py optuna --optuna-trials 50 --optuna-epochs 30

  # Prediction
  python main.py inference -M TwoTowerGAT -i data/test.csv
"""


def build_parser() -> argparse.ArgumentParser:
    """Build the headless CLI argument parser."""

    common = argparse.ArgumentParser(add_help=False)

    parser = argparse.ArgumentParser(
        prog="pharmagen",
        description="Pharmagen CLI Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EPILOG,
    )
    common.add_argument(
        *["-M", "--ml-model"],
        default="TwoTowerGAT",
        help="Model name for training/prediction (default: TwoTowerGAT)",
    )
    common.add_argument(
        *["-i", "--input"],
        type=Path,
        default=_DEFAULT_INPUT,
        help=f"Input data file path, CSV/TSV (default: {_DEFAULT_INPUT})",
    )
    common.add_argument(
        *["-p", "--epochs"],
        type=int,
        default=100,
        metavar="N",
        help="Number of training epochs (default: 100)",
    )
    common.add_argument(
        *["-v", "--verbose"], action="store_true", help="Enable verbose output"
    )
    common.add_argument(
        *["-d", "--debug"], action="store_true", help="Enable debug output"
    )

    subparsers = parser.add_subparsers(dest="mode", required=True)

    optuna_p = subparsers.add_parser(
        "optuna", parents=[common], help="Hyperparameter optimization with Optuna"
    )
    optuna_p.add_argument(
        "--optuna-trials",
        type=int,
        default=100,
        metavar="N",
        help="Number of Optuna trials (default: 100)",
    )
    optuna_p.add_argument(
        "--optuna-epochs",
        type=int,
        default=50,
        metavar="N",
        help="Number of epochs per Optuna trial (default: 50)",
    )
    # aquí añades las sub-opciones nuevas, ej:
    optuna_p.add_argument(
        "--optuna-sampler", choices=["tpe", "random", "cmaes"], default="tpe"
    )
    optuna_p.add_argument(
        "--optuna-pruner", choices=["median", "none"], default="median"
    )
    optuna_p.add_argument(
        "--optuna-storage", default=None, help="Optuna storage URL (sqlite/redis)"
    )
    optuna_p.add_argument("--optuna-study-name", default=None)
    optuna_p.add_argument(
        "--optuna-direction", choices=["minimize", "maximize"], default="minimize"
    )
    optuna_p.add_argument("--optuna-n-jobs", type=int, default=1)

    train_p = subparsers.add_parser("train", parents=[common], help="Train a model")
    inference_p = subparsers.add_parser(
        "inference", parents=[common], help="Run inference"
    )

    return parser
