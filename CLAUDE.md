# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Pharmagen** is a pharmacogenetic prediction tool that maps a patient's genomic profile (gene/allele) and prescribed medications to phenotypic outcomes, direction-of-effect, and adverse-drug-reaction types. The active architecture is a **Two-Tower Graph Neural Network** built on **GATv2** (PyTorch Geometric):

- **Drug tower** — molecular graphs derived from SMILES via RDKit (atomic + bond features).
- **Genotype tower** — variant topology graphs built from VCFs/TSVs validated against GRCh38.
- Towers are encoded with `GATv2Tower` (see `src/modeling/architectures/gnn.py`) and fused into multi-task heads sized by `target_dims`.

Earlier DeepFM code is referenced in places but is being phased out in favor of the GNN architecture.

## Environment & Commands

The project is managed with **uv** (`uv.lock` is the source of truth). Python is pinned to **3.14** in `.python-version` and `pyproject.toml`, but `ruff`/`mypy` still target `py310` — when adjusting tooling, expect this discrepancy.

```bash
# Install (editable, dev extras)
uv sync --extra dev

# Run the app — interactive menu (default)
python main.py

# Headless training
python main.py --mode train --model TwoTowerGAT --input train_data/train_data.tsv --epochs 100

# Optuna hyperparameter search
python main.py --mode train --optuna --optuna-trials 50 --optuna-epochs 30

# Headless prediction (writes <stem>_predictions_<date>.csv next to the input)
python main.py --mode predict --model TwoTowerGAT --input data/test.csv

# Lint / format / type-check
ruff check .
ruff format .
mypy src

# Tests (pytest discovers under tests/, see tests/pytest.ini)
pytest tests/
pytest tests/unit/
pytest tests/integration/
pytest tests/ -m "not slow"          # skip slow
pytest tests/ -m "not cuda"          # skip GPU-only
pytest tests/unit/config/test_manager.py::TestName::test_case  # single test
pytest tests/ --cov=src --cov-report=html
```

Note: `pyproject.toml`'s `[tool.pytest.ini_options]` and `tests/pytest.ini` disagree (the former adds `--cov=pharmagen` which doesn't exist as a package). Tests are typically driven by `tests/pytest.ini`; if a `pytest` invocation fails on coverage flags, run from `tests/` or override `--cov=src`.

## Repository State — Mid-Refactor

The current branch (`main`, ahead of commit `6644538 "Commit before refactor"`) is **mid-restructure**. Several modules imported by `src/pipeline.py` have been deleted from `src/utils/` but not yet replaced:

- Deleted: `src/utils/{checkpoint,data_utils,io,library_creator,library_creator_polars,losses,memory,metrics,module_builder,system,validation}.py`
- New (untracked): `src/types/`, `src/data/lib_builder_polars.py`, `src/data/lib_builder_v2.py`, `src/interface/io.py`, `docs/LIBRARY_BUILDER.md`, `src/CAJON DE SASTRE/` (a scratch/junk drawer — avoid editing).

Consequence: `from src.pipeline import train_pipeline` will currently fail at import time. When working in the pipeline/training path, expect to re-wire imports to the new homes (likely under `src/data/`, `src/types/`, and a future `src/utils/` rebuild) rather than restoring the deleted files. Verify what exists before assuming an import path is valid.

## Architecture

### Configuration (`src/config/`)
- `manager.py` is the canonical entry — it loads three TOMLs at import time (`paths.toml`, `settings.toml`, `models.toml`), exposes `DIRS`, `SEED`, `MULTI_LABEL_COLS`, `PROJECT_ROOT`, and `get_model_config(model_name)`. It validates everything against `ConfigValidator` (currently in the deleted `src/utils/validation.py` — see refactor note).
- `loader.py` defines an alternative `ModelConfigLoader` class but `manager.get_model_config` is what the rest of the code calls.
- Model definitions live in `src/config/models.toml`. Each model (e.g. `TwoTowerGAT`) declares `cols`, `features`, `targets`, dimension hints, and two parameter blocks: `[Model.params]` (active hyperparameters) and `[Model.optuna]` (search space, with values like `["log", 1e-4, 1e-2]`, `["categorical", 64, 128]`, `["int", 4, 6]`).

### Training Path (`src/pipeline.py` → `src/modeling/engine/`)
`train_pipeline` orchestrates: load+validate config → load+validate data → split → estimate memory → build `DoubleTowerDataset` (validation reuses train's `encoders` — critical) → infer real `drug_dim`/`haplo_dim`/`target_dims` from a sample → build `DoubleTowerCollater` DataLoaders → `create_gnn_model(...)` → `PGenTrainer.fit(...)`. Optuna goes through `src/modeling/engine/tuner.py::run_optuna_study`, and `PGenTrainer` has an `from_optuna` mode that disables checkpointing/compilation.

### Data (`src/data/`)
- `datasets.py` — `DoubleTowerDataset` returns dicts with `drug_data` and `haplo_data` PyG `Data` objects.
- `collator.py` — `DoubleTowerCollater` batches the two towers into a single batch.
- `lib_builder_polars.py` / `lib_builder_v2.py` — graph-library builders (drugs from SMILES, variants from VCF/TSV against `data/ref_genome/HSapiens_GChr38.fa`). See `docs/LIBRARY_BUILDER.md` for input schemas.

### Genomics (`src/genomics/`)
VCF handling, reference genome wrappers, and variant validation. Variants must be **1-based**; `start_pos` is decremented internally before FASTA lookup. Mismatches typically come from chromosome naming (`chr1` vs `NC_000001`) — `GLOBAL_CHROM_MAPPING` handles this.

### Interface (`src/interface/`)
- `cli.py` — interactive menu (`main_menu_loop`), uses lazy imports (`_get_train_pipeline`, etc.) to keep startup fast.
- `ui.py` — `ConsoleIO`, `Spinner`, `ProgressBar` primitives. Use these instead of bare `print` for user-facing output.
- `io.py` — license/notice printing.

### Outputs / Artifacts
Everything is rooted at `PROJECT_ROOT` (3 levels up from `src/config/manager.py`). Models save to `src/pgen_model/models/`, encoders to `src/pgen_model/encoders/`, Optuna reports to `reports/optuna_reports/`, training reports to `reports/train_reports/`, logs to `logs/`. These are auto-created on import of `manager.py`.

## Conventions

- New columns/targets must be added to **both** `models.toml` (`cols`, `features`, `targets`) **and** the relevant dataset/encoder logic — `train_dataset.encoders` is reused by validation, so anything missing there breaks dimension inference at `_infer_dimensions`.
- For multi-label targets, append the column name to `[project].multi_label_cols` in `settings.toml`; it flows through as `MULTI_LABEL_COLS`.
- Errors raised from training/data/config code should use the typed exception hierarchy in `src/utils/exceptions.py` (`ConfigurationError`, `DataError`, `ModelError`, `MemoryError`/`PharmagenMemoryError`, etc.) rather than bare `ValueError`/`RuntimeError`.
- Data files (`data/`, `train_data/`, `src/library/`, model artifacts) are gitignored and excluded from AI context via `.aiexclude` — don't expect them to be readable in fresh checkouts.
- Don't edit anything under `src/CAJON DE SASTRE/` — it's a scratch directory.
