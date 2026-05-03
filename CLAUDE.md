# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Pharmagen** is a pharmacogenetic prediction tool that maps a patient's genomic profile (gene/allele) and prescribed medications to phenotypic outcomes, direction-of-effect, and adverse-drug-reaction types. The active architecture is a **Two-Tower Graph Neural Network** built on **GATv2** (PyTorch Geometric):

- **Drug tower** — molecular graphs derived from SMILES via RDKit (atomic + bond features).
- **Genotype tower** — variant topology graphs built from VCFs/TSVs validated against GRCh38.
- Towers are encoded with `GATv2Tower` (`src/modeling/architectures/gnn.py`) and fused into multi-task heads sized by `target_dims`.

DeepFM code referenced in some places is being phased out in favor of the GNN.

## Refactor Status

This codebase is mid-refactor. The current plan and progress are in **`Ref.md`**. As of the last commit, Phases 0–3, 4a, 6 (partial), and 7 are complete:

- ✅ **Phase 0–1** — `src/CAJON DE SASTRE/` moved to `BACKUPS/`; broken imports restored under `src/utils/`.
- ✅ **Phase 2** — Pydantic v2 domain models in `src/domain/` (Drug, Variant, Gene, StarAllele, Position, Genotype, GraphMetadata, PredictionRequest/Result). 66 tests.
- ✅ **Phase 3** — Pydantic Settings in `src/config/`. `Settings` + `ModelConfig` (with discriminated `OptunaSpec` union for `["log", lo, hi]` / `["int", …]` / etc.). TOMLs moved to `src/config/data/`. `src/config/manager.py` is now a back-compat shim that exposes `DIRS`, `SEED`, `MULTI_LABEL_COLS`, etc. derived from the typed Settings. 25 tests.
- ✅ **Phase 4a** — Star-allele table extracted from `src/interface/io.py` to `data/dicts/star_alleles.tsv`, loaded via `src/genomics/star_alleles.py::StarAlleleMap`. 16 tests.
- ✅ **Phase 4.5** — Library builder rewritten as `src/data/library/` package (drugs, genes, pgx, chromosome, manifest, organize, config, builder, `__main__`). Replaces `lib_builder_polars.py` (883 LOC, module globals, shell scripts) and the abandoned `lib_builder_v2.py` stub. CLI: `python -m src.data.library`. Resume support via `build_manifest.json`. 47 tests; consumer contract (5105 drugs, 2353 gene variants) preserved.
- ⏳ **Phase 4b–f** (deferred) — split `DoubleTowerDataset`, `DataLoaderUtils`, `PGenTrainer` into focused units. The current `src/data/datasets.py` is still a god object.
- ✅ **Phase 5 (partial)** — `subprocess.run(shell=True, ...)` removed from `src/genomics/ngs_pipeline.py`; `bwa mem | samtools sort` runs via Python `Popen` plumbing. `src/genomics/variant_val.py::iter_variants` uses pysam for full VCF iteration with build-mismatch detection. `input()` calls removed from library code.
- ✅ **Phase 6 (partial)** — Spanish translated in `src/genomics/{ref_genome,ngs_pipeline,variant_val}.py`, `src/utils/logger.py`, `src/modeling/architectures/gnn.py`. `MemoryError` renamed to `PharmagenMemoryError` (with alias for back-compat); `ValidationError` no longer multi-inherits `IndexError, ValueError`. **Spanish remains in `src/data/datasets.py` and `src/data/lib_builder_polars.py`** — those are user-WIP files left for Phase 4.
- ✅ **Phase 7** — FastAPI inference service under `src/api/`. 19 tests.
- ⏳ **Phase 8** — CI workflow + docs sweep.

## Layout

```
src/
├── api/                 # FastAPI service (Phase 7)
│   ├── main.py          # create_app() factory + lifespan
│   ├── deps.py          # PredictorRegistry + DI
│   ├── schemas.py       # HTTP envelopes
│   └── routers/         # health, models, predict, library
├── domain/              # Pydantic v2 domain models (Phase 2)
│   ├── drug.py          # Drug + Drug.from_smiles()
│   ├── variant.py       # GenomeBuild, Position, Variant, Genotype
│   ├── gene.py          # Gene, StarAllele, AlleleFunction
│   ├── graph.py         # GraphKind, GraphMetadata, GraphPair
│   └── prediction.py    # PredictionRequest, PredictionResult
├── config/              # Pydantic Settings (Phase 3)
│   ├── __init__.py      # Public API: get_settings, get_model_config, …
│   ├── settings.py      # Settings (env-overridable via PHARMAGEN_*)
│   ├── models.py        # ModelConfig + OptunaSpec discriminated union
│   ├── paths.py         # Paths model with ensure_dirs()
│   ├── manager.py       # back-compat shim (DIRS, SEED, …)
│   └── data/            # paths.toml, settings.toml, models.toml
├── genomics/
│   ├── star_alleles.py  # StarAlleleMap from data/dicts/ (Phase 4a)
│   ├── reference.py     # was ref_genome.py — translated, no shell=True
│   ├── ngs_pipeline.py  # 4-phase NGS pipeline, argv-based subprocess
│   └── variant_val.py   # iter_variants() with build-mismatch validation
├── modeling/            # Largely unchanged; trainer/predictor still use legacy shim
├── data/
│   ├── datasets.py      # ⚠️ god object — Phase 4 will split
│   ├── collator.py
│   ├── graph_indexing.py
│   └── library/         # offline graph builder (Phase 4.5)
│       ├── builder.py   # LibraryBuilder orchestrator
│       ├── config.py    # LibraryBuildConfig (Pydantic)
│       ├── drugs.py     # smiles_to_graph + DrugGraphBuilder (25/7 schema)
│       ├── genes.py     # GenomicGraphBuilder (9/3 schema, FASTA validation)
│       ├── pgx.py       # PharmVar VCF folder loader
│       ├── chromosome.py # CHROM ↔ RefSeq map
│       ├── manifest.py  # resume tracking
│       ├── organize.py  # pure-Python file organization
│       └── __main__.py  # python -m src.data.library
├── interface/           # CLI + console utilities (Phase 4 will move into src/cli/)
├── utils/               # exceptions, logger, restored memory/validation/etc.
└── pipeline.py          # train_pipeline orchestrator
```

## Environment & Commands

The project is managed with **uv**. Python is pinned to **3.14**; ruff/mypy still target `py310` (a discrepancy worth fixing in Phase 8).

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

# Tests (use --override-ini to skip the addopts coverage flag in pyproject.toml,
# which references a "pharmagen" package that doesn't exist)
python -m pytest tests/unit/ -q --override-ini="addopts="
python -m pytest tests/unit/api/ -q --override-ini="addopts="
python -m pytest tests/unit/domain/ -v --override-ini="addopts="
```

## Conventions to keep

- **Pydantic everywhere at boundaries.** New code accepting external input (HTTP, CLI args, TSV rows, TOML) goes through `src/domain/` or `src/config/` models — `dict[str, Any]` is banned in those layers.
- **Coordinates are 1-based.** `Position.pos` matches FASTA/VCF conventions. Anything 0-based is internal-only and named `_0based`.
- **Genome build is part of every Position.** Build mismatch is a fail-fast error (`BioinformaticsError`), not a warning. See `iter_variants` for the pattern.
- **Star alleles come from `data/dicts/star_alleles.tsv`.** Don't reintroduce hardcoded tables in code. Add a row to the TSV instead.
- **No `shell=True` in subprocess calls.** Use argv lists. For pipes, use `subprocess.Popen` plumbing (see `ngs_pipeline.MappingAlignmentAnalysis.map_reads`).
- **No `input()` in library code.** Interactive prompts live in `src/interface/cli.py` or future `src/cli/workflows/`. Library functions take parameters or return generators.
- **Use `get_settings()` / `get_model_config()` in new code**, not `from src.config.manager import DIRS, SEED, …`. The shim exists for legacy callers and will be removed in a follow-up phase.
- **No emoji in log messages.** Emoji are for `ConsoleIO` user-facing output. Log messages use `logger.info("Doing X (sample=%s)", sample_id)` style.
- **English everywhere.** Spanish identifiers/comments in `src/` are tech debt to fix; `BACKUPS/` is exempted.

## Outstanding tech debt

- `src/data/datasets.py` is 820 lines and mixes loading/encoding/caching/validation. Phase 4 will split into `GraphCache`, `EncoderRegistry`, slim `DoubleTowerDataset`.
- `src/modeling/engine/trainer.py` has dual-mode logic (standard vs Optuna) gated by `from_optuna` flag. Phase 4f will split into `StandardTrainer` + `OptunaTrialTrainer` over a common `TrainingLoop` base.
- `src/data/lib_builder_v2.py` has `...` stubs in `molecule_graph_builder()` and `main()`. Either complete or remove.
- `src/genomics/vcf_handler/wrapper.py` (32 lines) calls a non-existent C++ binary. Either remove or guard behind `shutil.which("vcf_tool")`.
- `pyproject.toml` `[tool.pytest.ini_options].addopts` includes `--cov=pharmagen` which doesn't exist as a package. Override with `--override-ini="addopts="` until fixed.
- `tests/pytest.ini` has its own (older) coverage config. Both should be unified in Phase 8.
- `src/config/manager.py` calls `paths.ensure_dirs()` on import — that's a side effect for back-compat. New code should call `get_settings().paths.ensure_dirs()` explicitly (see `src/api/main.py::lifespan`).
