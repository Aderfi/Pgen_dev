"""Argument parser for the Pharmagen CLI.

Kept free of heavy imports so ``--help`` and parser tests stay fast.
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_MODEL = "TwoTowerGAT"
DEFAULT_INPUT = Path("train_data/train_data.tsv")
DEFAULT_FOLDS = 5  # mirrors src.evaluation.DEFAULT_FOLDS (kept import-free)
SCENARIOS = ("random", "pair", "pmid", "drug_cold", "variant_cold", "gene_cold")

_EPILOG = """\
Examples:
  python main.py train -i train_data/train_data.tsv -e 100
  python main.py train --neo4j                      # rows straight from the KG
  python main.py train -i data/library/kg/training_rows.parquet
  python main.py optuna --trials 50 --epochs 30
  python main.py predict -i data/patients.csv
  python main.py split --scenarios all --folds 5   # grouped k-fold scenarios
  python main.py train --scenario drug_cold --fold 0
  python main.py evaluate --scenario drug_cold      # every trained fold
  python main.py evaluate --compare reports/eval/A reports/eval/B
"""


class _Parser(argparse.ArgumentParser):
    """Adds cross-option checks argparse cannot express."""

    def parse_args(self, args=None, namespace=None):  # type: ignore[override]
        ns = super().parse_args(args, namespace)
        if ns.mode in ("train", "optuna") and (ns.scenario is None) != (
            ns.fold is None
        ):
            self.error("--scenario and --fold must be given together")
        return ns


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "-M",
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model section of models.toml (default: {DEFAULT_MODEL})",
    )
    verbosity = p.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-q", "--quiet", action="store_true", help="Only warnings and errors"
    )
    verbosity.add_argument(
        "-d", "--debug", action="store_true", help="Debug-level logging"
    )


def _add_source(p: argparse.ArgumentParser) -> None:
    source = p.add_mutually_exclusive_group()
    source.add_argument(
        "-i",
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Training rows: TSV/CSV/parquet (default: {DEFAULT_INPUT})",
    )
    source.add_argument(
        "--neo4j",
        action="store_true",
        help="Read training rows from the Neo4j knowledge graph (Settings.kg)",
    )


def _add_fold(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--scenario",
        choices=SCENARIOS,
        default=None,
        help="Train on a persisted split scenario (see 'split'); needs --fold",
    )
    p.add_argument(
        "--fold", type=int, default=None, metavar="K", help="Fold of --scenario"
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the ``pharmagen`` argument parser."""
    parser = _Parser(
        prog="pharmagen",
        description="Pharmagen — pharmacogenomic outcome prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EPILOG,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    train = sub.add_parser("train", help="Train a model")
    _add_common(train)
    _add_source(train)
    train.add_argument("-e", "--epochs", type=int, default=100, metavar="N")
    train.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=None,
        metavar="N",
        help="Default: params.batch_size from models.toml",
    )
    train.add_argument("--patience", type=int, default=10, metavar="N")
    train.add_argument("--val-split", type=float, default=0.2, metavar="F")
    _add_fold(train)

    optuna = sub.add_parser("optuna", help="Hyperparameter search with Optuna")
    _add_common(optuna)
    _add_source(optuna)
    optuna.add_argument("--trials", type=int, default=100, metavar="N")
    optuna.add_argument(
        "--epochs",
        type=int,
        default=None,
        metavar="N",
        help="Epochs per trial (default: optuna.epochs in models.toml)",
    )
    optuna.add_argument("--sampler", choices=["tpe", "random", "cmaes"], default="tpe")
    optuna.add_argument(
        "--pruner", choices=["hyperband", "median", "none"], default="hyperband"
    )
    optuna.add_argument(
        "--storage", default=None, help="Optuna storage URL (default: local sqlite)"
    )
    optuna.add_argument("--study-name", default=None)
    optuna.add_argument("--n-jobs", type=int, default=1, metavar="N")
    _add_fold(optuna)

    split = sub.add_parser(
        "split", help="Make the grouped k-fold evaluation split scenarios"
    )
    _add_common(split)
    split.add_argument(
        "-i",
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Training rows file the split is bound to (default: {DEFAULT_INPUT})",
    )
    split.add_argument(
        "--scenarios",
        default="all",
        help=f"Comma list of {', '.join(SCENARIOS)}, or 'all' (default)",
    )
    split.add_argument(
        "--folds",
        type=int,
        default=DEFAULT_FOLDS,
        metavar="K",
        help="Folds (default 5)",
    )

    predict = sub.add_parser("predict", help="Predict outcomes for a file of pairs")
    _add_common(predict)
    predict.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="CSV/TSV with the model features",
    )
    predict.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV (default: <input>_predictions_<date>.csv)",
    )

    evaluate = sub.add_parser(
        "evaluate", help="Evaluate trained scenario folds and write the HTML report"
    )
    _add_common(evaluate)
    evaluate.add_argument(
        "-i",
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Training rows file the split is bound to (default: {DEFAULT_INPUT})",
    )
    what = evaluate.add_mutually_exclusive_group(required=True)
    what.add_argument("--scenario", choices=SCENARIOS, help="Scenario to evaluate")
    what.add_argument(
        "--compare", nargs="+", type=Path, metavar="DIR", help="Compare evaluations"
    )
    what.add_argument(
        "--render", type=Path, metavar="DIR", help="Re-render report.html only"
    )
    evaluate.add_argument(
        "--folds",
        default=None,
        help="Comma list of folds (default: every trained fold)",
    )
    evaluate.add_argument(
        "--bootstrap", type=int, default=2000, metavar="N", help="Bootstrap resamples"
    )

    return parser
