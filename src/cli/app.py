"""CLI dispatch: configure logging, route ``args.mode`` to its handler.

Handlers import the heavy stack (torch, polars, optuna) lazily so ``--help``
stays instant.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import TYPE_CHECKING

from src.cli import console
from src.cli.parser import build_parser
from src.config import get_settings
from src.core import PharmagenException, setup_logging

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

Handler = Callable[[argparse.Namespace], None]


def _log_level(args: argparse.Namespace) -> int:
    if args.debug:
        return logging.DEBUG
    if args.quiet:
        return logging.WARNING
    return logging.INFO


def _source(args: argparse.Namespace) -> Path | str:
    """``"neo4j"`` or an existing input file (exit 1 otherwise)."""
    if getattr(args, "neo4j", False):
        return "neo4j"
    if not args.input.exists():
        console.error(f"Input file not found: {args.input}")
        sys.exit(1)
    return args.input


def _fold_ref(args: argparse.Namespace):
    """The ``FoldRef`` of ``--scenario/--fold``, or None."""
    if getattr(args, "scenario", None) is None:
        return None
    from src.evaluation.entities import Scenario
    from src.evaluation.runs import FoldRef

    return FoldRef(scenario=Scenario(args.scenario), fold=args.fold)


def _scenarios(spec: str):
    from src.evaluation.entities import Scenario

    if spec.strip() == "all":
        return list(Scenario)
    try:
        return [Scenario(s.strip()) for s in spec.split(",") if s.strip()]
    except ValueError as exc:
        console.error(f"Unknown scenario: {exc}")
        sys.exit(2)


def run_split(args: argparse.Namespace) -> None:
    from src.config import get_model_config
    from src.evaluation.entities import Scenario
    from src.evaluation.runs import create_splits
    from src.evaluation.splits import load_split

    source = _source(args)
    scenarios = _scenarios(args.scenarios)
    console.header(f"Splitting {source} into {args.folds} folds")
    for directory in create_splits(
        source, get_model_config(args.model), scenarios, k=args.folds
    ):
        _, manifest = load_split(Scenario(directory.name))
        tests = [f["test"] for f in manifest.rows_per_fold]
        console.success(
            f"{manifest.scenario}: test rows per fold {tests}, "
            f"never tested {manifest.rows_never_tested} -> {directory}"
        )


def run_train(args: argparse.Namespace) -> None:
    from src.pipeline import train_pipeline

    source = _source(args)
    split = _fold_ref(args)
    console.header(
        f"Training {args.model}"
        + (f" on {split.scenario} fold {split.fold}" if split else "")
    )
    console.info(f"Data: {source} | epochs: {args.epochs}")
    best = train_pipeline(
        model_name=args.model,
        source=source,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.val_split,
        patience=args.patience,
        split=split,
    )
    console.success(f"Training finished — best validation loss {best:.4f}")


def run_optuna(args: argparse.Namespace) -> None:
    from src.model.engine.tuner import run_optuna_study

    source = _source(args)
    console.header(f"Optuna search for {args.model}")
    console.info(f"Data: {source} | trials: {args.trials}")
    study = run_optuna_study(
        model_name=args.model,
        source=source,
        n_trials=args.trials,
        epochs=args.epochs,
        sampler=args.sampler,
        pruner=args.pruner,
        storage=args.storage,
        study_name=args.study_name,
        n_jobs=args.n_jobs,
        split=_fold_ref(args),
    )
    best = study.best_trial
    console.success(f"Best trial #{best.number}: {best.value:.4f} {best.params}")


def run_predict(args: argparse.Namespace) -> None:
    import polars as pl

    from src.model.engine.predictor import PGenPredictor

    if not args.input.exists():
        console.error(f"Input file not found: {args.input}")
        sys.exit(1)
    console.header(f"Predicting with {args.model}")
    try:
        predictor = PGenPredictor(args.model)
    except FileNotFoundError as exc:
        console.error(f"Model '{args.model}' is not trained: {exc}")
        console.info("Train it first: python main.py train")
        sys.exit(1)

    results = predictor.predict_file(args.input)
    if not results:
        console.warning("No predictions generated (see the log for details)")
        return
    out = args.output or args.input.with_name(
        f"{args.input.stem}_predictions_{datetime.now():%Y-%m-%d}.csv"
    )
    # Multi-label targets decode to lists; CSV cells need scalars.
    rows = [
        {k: "|".join(map(str, v)) if isinstance(v, list) else v for k, v in r.items()}
        for r in results
    ]
    pl.DataFrame(rows).write_csv(out)
    console.success(f"{len(results)} predictions saved to {out}")


def run_evaluate(args: argparse.Namespace) -> None:
    from src.evaluation.compare import compare
    from src.evaluation.entities import Scenario
    from src.evaluation.report import render
    from src.evaluation.runner import evaluate_scenario

    if args.render is not None:
        console.success(f"Report: {render(args.render)}")
        return
    if args.compare:
        out = compare(args.compare)
        console.success(f"Comparison: {out / 'report.html'}")
        return
    source = _source(args)
    folds = [int(f) for f in args.folds.split(",") if f.strip()] if args.folds else None
    console.header(f"Evaluating {args.model} on {args.scenario}")
    out = evaluate_scenario(
        args.model,
        source,
        Scenario(args.scenario),
        folds=folds,
        n_boot=args.bootstrap,
    )
    console.success(f"Report: {render(out)}")


_DISPATCH: dict[str, Handler] = {
    "train": run_train,
    "optuna": run_optuna,
    "predict": run_predict,
    "split": run_split,
    "evaluate": run_evaluate,
}


def run(args: argparse.Namespace) -> None:
    """Configure logging and dispatch to the handler for ``args.mode``."""
    get_settings().paths.ensure_dirs()
    setup_logging(level=_log_level(args))
    try:
        _DISPATCH[args.mode](args)
    except KeyboardInterrupt:
        console.warning("Cancelled by user")
        sys.exit(130)
    except PharmagenException as exc:
        logger.error("%s failed: %s", args.mode, exc)
        console.error(str(exc))
        sys.exit(1)
    except Exception as exc:
        logger.critical("Unhandled error in %s", args.mode, exc_info=True)
        console.error(f"Unexpected error: {exc}")
        console.info(f"Details in the log: {get_settings().paths.logs}")
        sys.exit(1)


def main(argv: Sequence[str] | None = None) -> None:
    """Parse ``argv`` and run."""
    run(build_parser().parse_args(argv))
