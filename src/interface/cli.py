# src/interface/cli.py
# Pharmagen - Command Line Interface
# Interactive Menu and Workflows

import datetime
import logging
import sys
from pathlib import Path

# Project Imports
from src.config import PROJECT_ROOT, get_available_models, get_settings
from src.interface.io import (
    print_conditions_details,
    print_gnu_notice,
    print_warranty_details,
)
from src.interface.ui import ConsoleIO, Spinner

logger = logging.getLogger(__name__)
DATE_STAMP = datetime.datetime.now().strftime("%Y_%m_%d")

# ==============================================================================
# MODULE CACHE - Lazy Imports
# ==============================================================================


def _get_train_pipeline():
    """Lazy import with automatic cache of train_pipeline."""
    from src.pipeline import train_pipeline

    return train_pipeline


def _get_optuna_study():
    """Lazy import with automatic cache of run_optuna_study."""
    from src.model.engine.tuner import run_optuna_study

    return run_optuna_study


def _get_predictor_class():
    """Lazy import with automatic cache of PGenPredictor."""
    from src.model.engine.predictor import PGenPredictor

    return PGenPredictor


# ==============================================================================
# UTILS
# ==============================================================================


def _select_model() -> str:
    """Interactively select a model from configuration."""
    models = get_available_models()
    if not models:
        ConsoleIO.print_error("No models found in models. toml")
        sys.exit(1)

    ConsoleIO.print_divider()
    ConsoleIO.print_info("Available Models:")
    for i, m in enumerate(models, 1):
        print(f"  {i}. {m}")
    ConsoleIO.print_divider()

    model_idx = ConsoleIO.input_int("Select model", min_val=1, max_val=len(models))
    return models[model_idx - 1]


# ==============================================================================
# INTERACTIVE CLI MENU WORKFLOWS
# ==============================================================================


def run_genomic_processing():
    """Simulation of Genomic ETL."""
    ConsoleIO.print_header("Genomic Processing Module")
    ConsoleIO.print_warning("NOT IMPLEMENTED YET")
    ConsoleIO.print_info("This module will process VCF files and genomic data")
    return  # noqa: PLR1711


def run_training_flow():
    """Interactive Training Workflow."""
    ConsoleIO.print_header("Training Module")

    # 1. Select Model
    model_name = _select_model()
    ConsoleIO.print_success(f"Selected model: {model_name}")

    # 2. Select Data File
    default_data = get_settings().paths.data / "processed" / "train_data. tsv"
    if not default_data.exists():
        # Fallback to project root default if exists
        fallback = PROJECT_ROOT / "train_data" / "train_data.tsv"
        if fallback.exists():
            default_data = fallback
        else:
            default_data = None

    csv_path = ConsoleIO.input_path(
        "Training Data Path", default=default_data, file_extensions=[". csv", ".tsv"]
    )

    # 3. Select Training Mode
    ConsoleIO.print_divider()
    mode = ConsoleIO.input_choice(
        "Select training mode", choices=["Standard", "Optuna"], default="Standard"
    )

    if mode == "Standard":
        _run_standard_training(model_name, csv_path)
    else:  # Optuna
        _run_optuna_training(model_name, csv_path)


def _run_standard_training(model_name: str, csv_path: Path):
    """Execute standard training workflow with cached import."""
    train_pipeline = _get_train_pipeline()

    # Get training parameters
    epochs = ConsoleIO.input_int(
        "Number of epochs", default=100, min_val=1, max_val=10000
    )

    # Confirmation
    ConsoleIO.print_divider()
    ConsoleIO.print_info(f"Model: {model_name}")
    ConsoleIO.print_info(f"Data: {csv_path}")
    ConsoleIO.print_info(f"Epochs:  {epochs}")

    if not ConsoleIO.confirm("Start training?", default=True):
        ConsoleIO.print_warning("Training cancelled")
        return

    # Execute training
    ConsoleIO.print_header(f"Training:  {model_name}")
    logger.info(f"Starting standard training: {model_name}")

    try:
        train_pipeline(model_name=model_name, csv_path=str(csv_path), epochs=epochs)
        ConsoleIO.print_success("Training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        ConsoleIO.print_error(f"Training failed: {e}")


def _run_optuna_training(model_name: str, csv_path: Path):
    """Execute Optuna optimization workflow with cached import."""
    run_optuna_study = _get_optuna_study()

    # Get Optuna parameters
    n_trials = ConsoleIO.input_int(
        "Number of trials", default=50, min_val=1, max_val=1000
    )

    epochs_per_trial = ConsoleIO.input_int(
        "Epochs per trial", default=50, min_val=1, max_val=500
    )

    # Confirmation
    ConsoleIO.print_divider()
    ConsoleIO.print_info(f"Model: {model_name}")
    ConsoleIO.print_info(f"Data: {csv_path}")
    ConsoleIO.print_info(f"Trials: {n_trials}")
    ConsoleIO.print_info(f"Epochs per trial: {epochs_per_trial}")

    if not ConsoleIO.confirm("Start Optuna optimization?", default=True):
        ConsoleIO.print_warning("Optimization cancelled")
        return

    # Execute Optuna study
    ConsoleIO.print_header(f"Optuna Optimization: {model_name}")
    logger.info(f"Starting Optuna study: {model_name}")

    try:
        run_optuna_study(
            model_name=model_name,
            csv_path=csv_path,
            n_trials=n_trials,
            epochs=epochs_per_trial,
        )
        ConsoleIO.print_success("Optuna optimization completed!")
    except Exception as e:
        logger.error(f"Optuna optimization failed: {e}", exc_info=True)
        ConsoleIO.print_error(f"Optimization failed: {e}")


def run_prediction_flow():
    """Interactive Prediction Workflow with cached import."""
    PGenPredictor = _get_predictor_class()

    ConsoleIO.print_header("Prediction Module")

    # Select model
    model_name = _select_model()

    try:
        # Load model
        with Spinner("Loading model...", style="braille"):
            predictor = PGenPredictor(model_name)
        ConsoleIO.print_success(f"Model '{model_name}' loaded successfully")

        # Prediction submenu loop
        while True:
            ConsoleIO.print_divider()
            print("Prediction Options:")
            print("  1. Interactive (Single Prediction)")
            print("  2. Batch (File Prediction)")
            print("  3. Back to Main Menu")
            ConsoleIO.print_divider()

            sub_choice = ConsoleIO.input_choice(
                "Select option", choices=["1", "2", "3"], default="1"
            )

            if sub_choice == "1":
                _interactive_predict_loop(predictor)
            elif sub_choice == "2":
                _batch_predict_flow(predictor)
            else:  # "3"
                break

    except FileNotFoundError as e:
        logger.error(f"Model loading failed: {e}")
        ConsoleIO.print_error(f"Could not load model: {e}")
        ConsoleIO.print_info("Tip: Train the model first using the Training menu")
    except Exception as e:
        logger.error(f"Critical prediction error: {e}", exc_info=True)
        ConsoleIO.print_error(f"Unexpected error: {e}")


def _interactive_predict_loop(predictor):
    """Single prediction interactive loop."""
    ConsoleIO.print_header("Interactive Prediction")
    ConsoleIO.print_info("Enter feature values (type 'q' to cancel)")

    inputs = {}

    # Collect feature values
    for feature in predictor.feature_cols:
        val = input(f"  {feature}:  ").strip()
        if val.lower() == "q":
            ConsoleIO.print_warning("Prediction cancelled")
            return
        inputs[feature] = val

    # Run prediction
    with Spinner("Calculating prediction...", style="dots"):
        result = predictor.predict_single(inputs)

    # Display results
    if result:
        ConsoleIO.print_divider()
        ConsoleIO.print_success("Prediction Results:")
        for k, v in result.items():
            print(f"  🔹 {k}: {v}")
        ConsoleIO.print_divider()
    else:
        ConsoleIO.print_error("Prediction failed - check input values")


def _batch_predict_flow(predictor):
    """Batch prediction from file."""
    import pandas as pd

    ConsoleIO.print_header("Batch Prediction")

    # Get input file
    input_path = ConsoleIO.input_path(
        "Input file path", file_extensions=[".csv", ".tsv"]
    )

    # Run predictions
    with Spinner(f"Processing {input_path.name}.. .", style="braille"):
        results = predictor.predict_file(input_path)

    if not results:
        ConsoleIO.print_warning("No predictions generated")
        return

    # Save results
    out_path = input_path.parent / f"{input_path.stem}_predictions_{DATE_STAMP}.csv"

    results_df = pd.DataFrame(results)
    results_df.to_csv(out_path, index=False)

    ConsoleIO.print_success(f"Predictions saved to: {out_path}")
    ConsoleIO.print_info(f"Total predictions: {len(results)}")


def run_advanced_analysis():
    """Advanced analysis workflow (placeholder)."""
    ConsoleIO.print_header("Advanced Analysis")
    ConsoleIO.print_warning("NOT IMPLEMENTED YET")
    ConsoleIO.print_info("This module will provide:")
    print("  • Model interpretability reports")
    print("  • Feature importance analysis")
    print("  • SHAP value visualizations")
    print("  • Performance metrics dashboard")


# ==============================================================================
# MAIN MENU LOOP
# ==============================================================================


def main_menu_loop():
    """Main interactive menu loop."""
    logger.info("Starting interactive menu")
    print_gnu_notice()

    while True:
        ConsoleIO.print_header("Pharmagen - Main Menu")
        print("  1. Genomic Processing (ETL)")
        print("  2. Train Models (Deep Learning)")
        print("  3. Predict (Inference)")
        print("  4. Advanced Analysis")
        print("  5. Exit")
        print()
        print("  Type 'show w' for warranty details")
        print("  Type 'show c' for license conditions")
        ConsoleIO.print_divider("=")

        choice = input("Select option (1-5): ").strip()

        # Easter eggs for license info
        if choice == "show w":
            print_warranty_details()
            continue
        if choice == "show c":
            print_conditions_details()
            continue

        # Main menu options
        if choice == "1":
            run_genomic_processing()
        elif choice == "2":
            run_training_flow()
        elif choice == "3":
            run_prediction_flow()
        elif choice == "4":
            run_advanced_analysis()
        elif choice == "5":
            if ConsoleIO.confirm("Are you sure you want to exit?", default=False):
                logger.info("User exit")
                ConsoleIO.print_info("Goodbye!  👋")
                sys.exit(0)
        else:
            ConsoleIO.print_error("Invalid option - please select 1-5")


if __name__ == "__main__":
    main_menu_loop()
