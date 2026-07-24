# src/tui/app.py
# Pharmagen - Headless CLI dispatch

"""Headless execution handlers and top-level dispatch.

``run`` configures logging, routes an parsed ``argparse.Namespace`` to the
matching handler, and centralizes interrupt/error handling. Handlers keep
heavy imports (pipeline, predictor, polars) local so ``--help`` stays cheap.
"""

import argparse
import logging
import sys
from collections.abc import Callable
from datetime import datetime

from src.config import get_settings
from src.core import setup_logging
from src.interface.ui import ConsoleIO, Spinner

logger = logging.getLogger("Pharmagen")

Handler = Callable[[argparse.Namespace], None]


def _configure_logging(args: argparse.Namespace) -> None:
    """Set the global logging level from the verbosity flags."""
    if args.debug:
        level = logging.DEBUG
    elif args.verbose:
        level = logging.INFO
    else:
        level = logging.WARNING

    setup_logging(name="Pharmagen", level=level)
    logger.setLevel(level)


def _require_input(args: argparse.Namespace) -> None:
    """Exit with an error if the input file does not exist."""
    if not args.input.exists():
        ConsoleIO.print_error(f"Input file not found: {args.input}")
        sys.exit(1)


def run_training(args: argparse.Namespace) -> None:
    """Execute training (standard or Optuna) in headless mode."""
    _require_input(args)

    if args.optuna:
        from src.model.engine.tuner import run_optuna_study

        logger.info("Starting Optuna optimization (model=%s)", args.model)
        ConsoleIO.print_header("Optuna Hyperparameter Optimization")
        ConsoleIO.print_info(f"Model: {args.model}")
        ConsoleIO.print_info(f"Data: {args.input}")
        ConsoleIO.print_info(f"Trials: {args.optuna_trials}")
        ConsoleIO.print_info(f"Epochs per trial: {args.optuna_epochs}")

        run_optuna_study(
            model_name=args.model,
            csv_path=str(args.input),
            n_trials=args.optuna_trials,
            epochs=args.optuna_epochs,
        )

        ConsoleIO.print_success("Optuna optimization completed!")
        return

    from src.pipeline import train_pipeline

    logger.info("Starting standard training (model=%s)", args.model)
    ConsoleIO.print_header("Standard Training")
    ConsoleIO.print_info(f"Model: {args.model}")
    ConsoleIO.print_info(f"Data: {args.input}")
    ConsoleIO.print_info(f"Epochs: {args.epochs}")

    with Spinner(f"Training {args.model}...", style="braille"):
        train_pipeline(
            model_name=args.model, csv_path=str(args.input), epochs=args.epochs
        )

    ConsoleIO.print_success("Training completed successfully!")


def run_prediction(args: argparse.Namespace) -> None:
    """Execute prediction in headless mode."""
    import polars as pl

    from src.model.engine.predictor import PGenPredictor

    _require_input(args)

    logger.info("Starting headless prediction (model=%s)", args.model)
    ConsoleIO.print_header("Prediction Mode")
    ConsoleIO.print_info(f"Model: {args.model}")
    ConsoleIO.print_info(f"Input: {args.input}")

    try:
        with Spinner("Loading model...", style="braille"):
            predictor = PGenPredictor(args.model)
        ConsoleIO.print_success("Model loaded successfully")

        with Spinner(f"Processing {args.input.name}...", style="braille"):
            results = predictor.predict_file(args.input)

        if not results:
            ConsoleIO.print_warning("No predictions generated")
            return

        date_stamp = datetime.now().strftime("%Y-%m-%d")
        out_path = args.input.parent / f"{args.input.stem}_predictions_{date_stamp}.csv"

        pl.DataFrame(data=results).write_csv(out_path)

        ConsoleIO.print_success(f"Predictions saved to: {out_path}")
        ConsoleIO.print_info(f"Total predictions: {len(results)}")

    except FileNotFoundError as exc:
        logger.error("Model not found (model=%s): %s", args.model, exc)
        ConsoleIO.print_error(f"Model '{args.model}' not found")
        ConsoleIO.print_info("Tip: Train the model first using --mode train")
        sys.exit(1)


_DISPATCH: dict[str, Handler] = {
    "train": run_training,
    "predict": run_prediction,
}


def run(args: argparse.Namespace) -> None:
    """Configure logging and dispatch to the handler for ``args.mode``."""
    _configure_logging(args)

    handler = _DISPATCH.get(args.mode)
    if handler is None:  # pragma: no cover - guarded by parser choices
        ConsoleIO.print_error(f"Unknown mode: {args.mode}")
        sys.exit(2)

    try:
        handler(args)
    except KeyboardInterrupt:
        ConsoleIO.print_warning("\nOperation cancelled by user.")
        logger.info("User interrupted execution")
        sys.exit(0)
    except Exception as exc:
        logger.critical("Unhandled error in dispatch: %s", exc, exc_info=True)
        ConsoleIO.print_error(f"Critical system error: {exc}")
        ConsoleIO.print_info(f"Check logs for details: {get_settings().paths.logs}")
        sys.exit(1)
