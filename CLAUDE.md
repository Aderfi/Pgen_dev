# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Pharmagen** is a pharmacogenetic prediction tool that maps a patient's genomic profile (gene/allele) and prescribed medications to phenotypic outcomes, direction-of-effect, and adverse-drug-reaction types. The active architecture is a **Two-Tower Graph Neural Network** built on **GATv2** (PyTorch Geometric):

- **Drug tower** — molecular graphs derived from SMILES via RDKit (atomic + bond features).
- **Genotype tower** — variant topology graphs built from VCFs/TSVs validated against GRCh38.
- Towers are encoded with `GATv2Tower` (`src/model/architectures/gnn.py`) and fused into multi-task heads sized by `target_dims`.

## Refactor Status

The core refactor is complete. The original plan lives in **`Ref.md`**. Highlights:

- **Phases 0–7** complete: backups extracted, Pydantic domain models (`src/domain/`), Pydantic Settings (`src/config/`), star-allele table externalized, `src/data/library/` rebuilt, `DataLoaderUtils` split (`loaders`/`normalize`/`cleaning`), `DoubleTowerDataset` decomposed (`cache`/`encoders`), `PGenTrainer` split (`loop`/`standard`/`optuna_trainer`), `subprocess.run(shell=True, …)` purged, Spanish translated, FastAPI service under `src/api/`.
- **Post-refactor cleanup** — `src/modeling/` renamed to `src/model/`; `src/utils/` split into `src/core/{exceptions,log,validation}.py`; `src/config/manager.py` shim removed; the `vcf_handler/` package deleted; pre-refactor snapshot moved to `BACKUPS/dev_Pharmagen_snapshot/` (tag `pre-refactor-2026-05`).
- **Phase 8** complete: GitHub Actions CI (`.github/workflows/ci.yml`), integration smoke tests under `tests/integration/`, and the doc sweep that produced this file.
- **Engine consolidation** — `src/model/engine/base.py` now owns the device/data/dataset/loader/model bootstrap shared by training, tuning, and inference. `PGenPredictor` was rewritten on top of `DoubleTowerDataset` + `DoubleTowerCollater` + the GNN forward; the DeepFM-era LabelEncoder inference path is gone.
- **Library relocation** — graph artefacts moved from `src/library/` to `data/library/`, routed through `Settings.paths.library`. Callers no longer build `PROJECT_ROOT / "src" / "library"` by hand.

## Layout

```
src/
├── api/                 # FastAPI service (create_app + routers + DI)
├── core/                # Cross-cutting: exception hierarchy, logging, validators
├── config/              # Pydantic Settings (settings/models/paths + data/*.toml)
├── domain/              # Pydantic v2 domain models (Drug, Variant, Gene, …)
├── data/
│   ├── cache.py         # GraphCache + GraphDims
│   ├── cleaning.py      # GenoKeyBuilder + PharmacogenomicCleaner
│   ├── collator.py
│   ├── datasets.py      # Slim DoubleTowerDataset composing cache + encoder
│   ├── encoders.py      # TargetEncoder
│   ├── graph_indexing.py
│   ├── loaders.py       # TabularLoader
│   ├── normalize.py     # MultiLabelNormalizer + Stratifier
│   └── library/         # Offline graph builder (python -m src.data.library)
├── genomics/
│   ├── ngs_pipeline.py  # 4-phase NGS pipeline (argv-based subprocess)
│   ├── ref_genome.py    # Reference handling
│   ├── star_alleles.py  # StarAlleleMap (data/dicts/star_alleles.tsv)
│   └── variant_val.py   # iter_variants() with build-mismatch detection
├── interface/           # CLI + console utilities (ui.py, cli.py, io.py)
├── model/
│   ├── architectures/   # GATv2 layers + assembly (create_gnn_model)
│   ├── checkpoint.py    # CheckpointManager
│   ├── engine/
│   │   ├── base.py      # Shared bootstrap: device, data, datasets, loaders, model build
│   │   ├── predictor.py # PGenPredictor — GNN inference engine
│   │   └── tuner.py     # PGenTuner — Optuna search orchestrator
│   ├── factories.py     # LossFactory + OptimizerFactory
│   ├── losses.py        # MultiTaskUncertaintyLoss
│   └── training/        # TrainingLoop, StandardTrainer, OptunaTrialTrainer
└── pipeline.py          # train_pipeline orchestrator (delegates to engine.base)

data/
├── library/             # On-disk graph cache (.pt files for drugs/ + gene_graphs/)
├── dicts/               # star_alleles.tsv and other static lookups
├── raw/, processed/     # Training inputs
└── ref_genome/          # Reference FASTA + indices

scripts/                 # Standalone visualisation / inspection utilities
tests/                   # unit/, integration/, benchmarks/, fixtures/
.github/workflows/ci.yml # Ruff check + format check + pytest on push/PR
```

## Environment & Commands

The project is managed with **uv**. Python is pinned to **3.14** and ruff/mypy target `py314`.

```bash
uv sync --extra dev           # install
uv run ruff check . && uv run ruff format .

# CLI
python main.py                # interactive menu (default)
python main.py --mode train --model TwoTowerGAT --input data/train.tsv --epochs 100
python main.py --mode predict --model TwoTowerGAT --input data/test.csv

# FastAPI inference service
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
# → http://localhost:8000/docs (auto OpenAPI)
# → GET /health, GET /v1/models, POST /v1/predict (single + /batch),
#   GET /v1/library/{drugs,genes,genes/{symbol}}

# Tests — addopts in pyproject.toml uses --cov=src.
uv run pytest tests/unit -q
uv run pytest tests/integration -q
uv run pytest tests/unit/api -q
```

## Engine contract (training ↔ inference)

`src/model/engine/base.py` exposes the helpers every engine uses; build new engines on top of these instead of duplicating bootstrap.

| Helper | Purpose |
|---|---|
| `resolve_device(override=None)` | Pick CUDA or CPU; honors explicit override. |
| `extract_tower_dims(cfg)` | Build the nested `{drugs/geno: {features, edges, attrs}}` dim spec from `cfg.extras`. |
| `load_and_clean_data(csv_path, cfg)` | `TabularLoader` → `PharmacogenomicCleaner`, with column / missingness validation. |
| `stratified_split(df, val_split)` | Train/val split honoring the `_stratify` column when present. |
| `build_two_tower_datasets(...)` | Paired train/val `DoubleTowerDataset` with shared encoders. |
| `infer_dataset_dimensions(...)` | Probe a sample to learn `drug_dim`, `geno_dim`, `target_dims`. |
| `build_train_val_loaders(...)` | Standard `DataLoader` pair with project defaults (collator, workers, pin_memory). |
| `build_gnn_model(...)` | Wraps `create_gnn_model` with the inferred dims. |

### Training artifacts bundle

`pipeline.train_pipeline` persists a single artifact at `paths.encoders/encoders_{model_name}.pkl` after dimension inference:

```python
{
    "encoders": dict[str, LabelEncoder | MultiLabelBinarizer],
    "drug_dim": int,    # inferred from real graphs, NOT cfg defaults
    "geno_dim": int,
    "schema_version": 1,
}
```

`PGenPredictor._load_training_artifacts` understands the bundle and falls back (with a warning) to the legacy plain-dict format for older pickles. Always prefer the bundle — the dims are required to recreate a model whose `state_dict` matches the saved checkpoint.

## Conventions to keep

- **Pydantic everywhere at boundaries.** New code accepting external input (HTTP, CLI args, TSV rows, TOML) goes through `src/domain/` or `src/config/` models — `dict[str, Any]` is banned in those layers.
- **Coordinates are 1-based.** `Position.pos` matches FASTA/VCF conventions. Anything 0-based is internal-only and named `_0based`.
- **Genome build is part of every Position.** Build mismatch is a fail-fast error (`BioinformaticsError`), not a warning. See `iter_variants` for the pattern.
- **Star alleles come from `data/dicts/star_alleles.tsv`.** Don't reintroduce hardcoded tables in code. Add a row to the TSV instead.
- **No `shell=True` in subprocess calls.** Use argv lists. For pipes, use `subprocess.Popen` plumbing (see `ngs_pipeline.MappingAlignmentAnalysis.map_reads`).
- **No `input()` in library code.** Interactive prompts live in `src/interface/cli.py`. Library functions take parameters or return generators.
- **Use `get_settings()` / `get_model_config()` in new code.** The old `src.config.manager` shim has been removed.
- **Graph library lives at `Settings.paths.library`.** Never hardcode `PROJECT_ROOT / "src" / "library"` — it no longer exists.
- **Engine bootstrap goes through `src/model/engine/base.py`.** Don't reimplement device selection, dataset wiring, or `create_gnn_model` calls inside engines.
- **Exceptions, logging, validators live in `src/core/`.** Import via `from src.core import EncoderError`, `from src.core import setup_logging`, etc. — not the old flat-`src/` modules.
- **No emoji in log messages.** Emoji are for `ConsoleIO` user-facing output. Log messages use `logger.info("Doing X (sample=%s)", sample_id)` style.
- **English everywhere.** Spanish identifiers/comments in `src/` are tech debt to fix; `BACKUPS/` is exempted.

## Outstanding tech debt

- `main.py` (~300 LOC) still mixes CLI parsing, logging setup, and dispatch; a future `src/cli/app.py` would keep `main.py` as a one-liner entry point.
- End-to-end predictor verification still requires a real trained checkpoint; the integration smoke tests only cover the import + artifact-loading paths.
