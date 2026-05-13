# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Pharmagen** is a pharmacogenetic prediction tool that maps a patient's genomic profile (gene/allele) and prescribed medications to phenotypic outcomes, direction-of-effect, and adverse-drug-reaction types. The active architecture is a **Two-Tower Graph Neural Network** built on **GATv2** (PyTorch Geometric):

- **Drug tower** — molecular graphs derived from SMILES via RDKit (atomic + bond features).
- **Genotype tower** — variant topology graphs built from VCFs/TSVs validated against GRCh38.
- Towers are encoded with `GATv2Tower` (`src/modeling/architectures/gnn.py`) and fused into multi-task heads sized by `target_dims`.

DeepFM code referenced in some places is being phased out in favor of the GNN.

## Refactor Status

This codebase has completed the core refactor. The original plan is in **`Ref.md`**. Highlights:

- ✅ **Phases 0–7** complete: backups extracted, Pydantic domain models (`src/domain/`), Pydantic Settings (`src/config/`), star-allele table externalized, `src/data/library/` rebuilt, `DataLoaderUtils` split (`loaders`/`normalize`/`cleaning`), `DoubleTowerDataset` decomposed (`cache`/`encoders`), `PGenTrainer` split (`loop`/`standard`/`optuna_trainer`), `subprocess.run(shell=True, …)` purged, Spanish translated, FastAPI service under `src/api/`.
- ✅ **Post-refactor cleanup** — `src/modeling/` renamed to `src/model/`; `src/utils/` split into `src/core/{exceptions,log,validation}.py`; `src/config/manager.py` shim removed; the `vcf_handler/` package deleted (calling a non-existent C++ binary); `dev_Pharmagen/` pre-refactor snapshot moved to `BACKUPS/dev_Pharmagen_snapshot/` (also reachable via tag `pre-refactor-2026-05`); stray data artefacts and the duplicate `src/library_archive.tar.gz` removed.
- ⏳ **Phase 8 pending** — CI workflow and a final docs sweep.

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
├── library/             # On-disk graph cache: .pt files for drugs/ + gene_graphs/
├── model/               # Was src/modeling/ — renamed post-refactor
│   ├── architectures/   # GATv2 layers + assembly (create_gnn_model)
│   ├── checkpoint.py    # CheckpointManager
│   ├── engine/          # predictor + tuner
│   ├── factories.py     # LossFactory + OptimizerFactory
│   ├── losses.py        # MultiTaskUncertaintyLoss
│   └── training/        # TrainingLoop, StandardTrainer, OptunaTrialTrainer
└── pipeline.py          # train_pipeline orchestrator
```

## Environment & Commands

The project is managed with **uv**. Python is pinned to **3.14** and ruff/mypy target `py314`.

```bash
uv sync --extra dev           # install
ruff check . && ruff format .

# CLI
python main.py                # interactive menu (default)
python main.py --mode train --model TwoTowerGAT --input data/train.tsv --epochs 100
python main.py --mode predict --model TwoTowerGAT --input data/test.csv

# FastAPI inference service
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
# → http://localhost:8000/docs (auto OpenAPI)
# → GET /health, GET /v1/models, POST /v1/predict (single + /batch),
#   GET /v1/library/{drugs,genes,genes/{symbol}}

# Tests — addopts in pyproject.toml uses --cov=src now, so no override needed.
python -m pytest tests/unit/ -q
python -m pytest tests/unit/api/ -q
python -m pytest tests/unit/domain/ -v
```

## Conventions to keep

- **Pydantic everywhere at boundaries.** New code accepting external input (HTTP, CLI args, TSV rows, TOML) goes through `src/domain/` or `src/config/` models — `dict[str, Any]` is banned in those layers.
- **Coordinates are 1-based.** `Position.pos` matches FASTA/VCF conventions. Anything 0-based is internal-only and named `_0based`.
- **Genome build is part of every Position.** Build mismatch is a fail-fast error (`BioinformaticsError`), not a warning. See `iter_variants` for the pattern.
- **Star alleles come from `data/dicts/star_alleles.tsv`.** Don't reintroduce hardcoded tables in code. Add a row to the TSV instead.
- **No `shell=True` in subprocess calls.** Use argv lists. For pipes, use `subprocess.Popen` plumbing (see `ngs_pipeline.MappingAlignmentAnalysis.map_reads`).
- **No `input()` in library code.** Interactive prompts live in `src/interface/cli.py`. Library functions take parameters or return generators.
- **Use `get_settings()` / `get_model_config()` in new code.** The old `src.config.manager` shim has been removed.
- **Exceptions, logging, validators live in `src/core/`.** Import via `from src.core import EncoderError`, `from src.core import setup_logging`, etc. — not the old flat-`src/` modules.
- **No emoji in log messages.** Emoji are for `ConsoleIO` user-facing output. Log messages use `logger.info("Doing X (sample=%s)", sample_id)` style.
- **English everywhere.** Spanish identifiers/comments in `src/` are tech debt to fix; `BACKUPS/` is exempted.

## Outstanding tech debt

- `src/library/` still mixes generated `.pt` graph caches with the (now-empty) `__init__.py` package marker; tightly coupled to `src/api/routers/library.py`, `src/data/graph_indexing.py`, `src/data/datasets.py`. A future phase should relocate the artefacts to `data/library/` and turn `src/library` into pure code (or delete it).
- `src/data/datasets.py` still constructs its library root as `PROJECT_ROOT / "src" / "library"`. Once the artefacts move, switch to `get_settings().paths.<…>`.
- `src/model/engine/{predictor,tuner}.py` weren't restructured during the trainer split; they still own a lot of device/dataloader/encoder bootstrap and could share a base with `TrainingLoop`.
- `main.py` (~300 LOC) mixes CLI parsing, logging setup, and dispatch; a future `src/cli/app.py` would keep `main.py` as a one-liner entry point.
- No CI workflow under `.github/workflows/` yet — Phase 8 deliverable.
- `tests/integration/` is empty after the refactor; needs new smoke tests (pipeline import, predict round-trip).
