# Pharmagen Architecture

This document describes the runtime layout of `src/` after the 2026-Q2 refactor (see `Ref.md` for the phased history). It's the canonical reference for new contributors and for AI assistants working on the codebase.

## High-level picture

Pharmagen has four user-facing surfaces — all of them sit on top of the same domain models, configuration, and trained model artifacts:

```
                               ┌──────────────────┐
                               │   src/domain/    │  Pydantic v2 models
                               │  (single source  │  Drug, Variant, Gene,
                               │   of truth for   │  StarAllele, GraphMetadata,
                               │   bio + ML data) │  PredictionRequest, …
                               └────────┬─────────┘
                                        │ used by
       ┌────────────────────────────────┼────────────────────────────────┐
       │                                │                                │
┌──────▼──────┐  ┌──────────────┐  ┌────▼───────┐  ┌─────────────────────▼────┐
│ src/cli/    │  │ src/api/     │  │ src/data/  │  │ src/modeling/training/   │
│ (interactive│  │ (FastAPI     │  │ library/   │  │ (StandardTrainer +       │
│  menu, in   │  │  inference   │  │ (offline   │  │  OptunaTrialTrainer over │
│  src/       │  │  service)    │  │  graph     │  │  a TrainingLoop ABC)     │
│  interface/ │  │              │  │  builder)  │  │                          │
│  for now)   │  └──────┬───────┘  └────┬───────┘  └─────────────┬────────────┘
└──────┬──────┘         │               │                        │
       │                │               │                        │
       │                ▼               ▼                        ▼
       │          ┌──────────────────────────────────┐   ┌────────────────┐
       │          │ src/data/                        │   │ src/library/   │
       │          │   datasets.py (DoubleTowerDataset)│  │ drugs/*.pt     │
       │          │   cache.py    (GraphCache)        │  │ gene_graphs/   │
       │          │   encoders.py (TargetEncoder)     │  │   <gene>/*.pt  │
       │          │   loaders.py  (TabularLoader)     │  └───────▲────────┘
       │          │   cleaning.py (PharmacogenomicCleaner)       │
       │          │   normalize.py / collator.py / graph_indexing.py
       │          └──────────────────────────────────┘           │
       │                                                         │
       └─► src/pipeline.py orchestrates training; predictions read this on-disk library.
```

## Module map

```
src/
├── pipeline.py             Training orchestrator (train_pipeline)
│
├── api/                    FastAPI inference service (Phase 7)
│   ├── main.py             create_app() factory + lifespan
│   ├── deps.py             PredictorRegistry + DI helpers
│   ├── schemas.py          HTTP request / response envelopes
│   └── routers/            health · models · predict · library
│
├── cli/  (planned)         Reserved namespace for the interactive menu.
│                           Today the menu lives in src/interface/cli.py;
│                           Phase 9 will move it under src/cli/workflows/.
│
├── config/                 Pydantic Settings (Phase 3)
│   ├── __init__.py         Public API: get_settings, get_model_config, …
│   ├── settings.py         Settings (env-overridable via PHARMAGEN_*)
│   ├── paths.py            Paths model with explicit ensure_dirs()
│   ├── models.py           ModelConfig + OptunaSpec discriminated union
│   ├── manager.py          Back-compat shim (DIRS, SEED, MULTI_LABEL_COLS, …)
│   └── data/               TOML data files (paths, settings, models)
│
├── data/                   Data loading + preprocessing
│   ├── datasets.py         DoubleTowerDataset (slim, composes the below)
│   ├── cache.py            GraphCache + GraphDims + make_empty_graph
│   ├── encoders.py         TargetEncoder (single + multi-label)
│   ├── loaders.py          TabularLoader (CSV/TSV with project schema)
│   ├── cleaning.py         GenoKeyBuilder + PharmacogenomicCleaner
│   ├── normalize.py        MultiLabelNormalizer + Stratifier
│   ├── collator.py         DoubleTowerCollater (PyG batching)
│   ├── graph_indexing.py   GraphIndexBuilder (walks src/library/)
│   └── library/            Offline graph builder (Phase 4.5)
│       ├── builder.py      LibraryBuilder orchestrator
│       ├── config.py       LibraryBuildConfig (Pydantic)
│       ├── drugs.py        smiles_to_graph + DrugGraphBuilder (25/7 schema)
│       ├── genes.py        GenomicGraphBuilder (9/3 schema, FASTA validation)
│       ├── pgx.py          PharmVar VCF folder loader
│       ├── chromosome.py   CHROM ↔ RefSeq accession map
│       ├── manifest.py     Resume tracking (atomic JSON manifest)
│       ├── organize.py     Pure-Python file organization
│       └── __main__.py     CLI: python -m src.data.library
│
├── domain/                 Pydantic v2 domain models (Phase 2)
│   ├── drug.py             Drug + Drug.from_smiles()
│   ├── variant.py          GenomeBuild · Position · Variant · Genotype
│   ├── gene.py             Gene · StarAllele · AlleleFunction
│   ├── graph.py            GraphKind · GraphMetadata · GraphPair
│   └── prediction.py       PredictionRequest · PredictionResult
│
├── genomics/               Bioinformatics integrations
│   ├── reference.py        GRCh38 download + samtools/bwa indexing
│   ├── ngs_pipeline.py     FastQC → BWA → Picard → Freebayes → VEP
│   ├── variant_val.py      iter_variants() with build-mismatch detection
│   ├── star_alleles.py     StarAlleleMap (loads data/dicts/star_alleles.tsv)
│   └── vcf_handler/        VCF helpers
│
├── interface/              Console UI (CLI menu, IO helpers)
│   ├── cli.py              main_menu_loop
│   ├── ui.py               ConsoleIO · Spinner · ProgressBar
│   └── io.py               JSON helpers · GPL notices · DataLoaderUtils facade
│
├── modeling/
│   ├── architectures/      Towers + factory
│   │   ├── gnn.py          GATv2Tower · PharmagenTwoTower
│   │   └── layers.py       create_gnn_model
│   ├── training/           Phase 4d trainer split
│   │   ├── loop.py         TrainingLoop ABC (shared concerns)
│   │   ├── standard.py     StandardTrainer (compile + checkpoints + tqdm)
│   │   └── optuna_trainer.py  OptunaTrialTrainer (trial reporting + pruning)
│   └── engine/             Legacy entry points
│       ├── trainer.py      PGenTrainer factory (back-compat)
│       ├── tuner.py        Optuna study orchestration
│       └── predictor.py    PGenPredictor (loads weights + encoders)
│
└── utils/                  General-purpose
    ├── exceptions.py       PharmagenException hierarchy
    ├── logger.py           setup_logging
    ├── memory.py           MemoryMonitor + estimators
    ├── checkpoint.py       CheckpointManager
    ├── losses.py           MultiTaskUncertaintyLoss + focal/asymmetric
    ├── module_builder.py   OptimizerFactory + LossFactory
    ├── metrics.py          Per-task metric helpers
    ├── system.py           Environment + GPU detection
    ├── validation.py       ConfigValidator + DataValidator
    ├── pgen_types.py       Lightweight TypeAliases
    └── io.py               Shim re-exporting DataLoaderUtils
```

## Configuration flow

Settings are resolved once per process. The flow is:

```
src/config/data/settings.toml ┐
src/config/data/paths.toml    ├─►  get_settings() ─► Settings (cached)
PHARMAGEN_* env vars          ┘                       ├── paths     → Paths model
                                                      ├── seed
                                                      ├── version
                                                      └── multi_label_cols
src/config/data/models.toml ─►  get_model_config(name) ─► ModelConfig
                                                              ├── features
                                                              ├── targets
                                                              ├── params       (fixed)
                                                              ├── optuna       (search)
                                                              └── extras
```

Legacy code (e.g. `src/pipeline.py`) imports flat constants from `src.config.manager` (`DIRS`, `SEED`, `MULTI_LABEL_COLS`, `REF_GENOME_FASTA`). Those are derived from `Settings` and exist as a back-compat shim. **New code imports from `src.config` directly.**

## Data flow during training

```
data/snp_data_output.tsv ─┐
data/drugs_cid.tsv        ├─► python -m src.data.library  ─► src/library/{drugs,gene_graphs}/*.pt
data/haplotype_variants/  │
data/ref_genome/*.fa      ┘
                                                  │
train_data/train_data.tsv ─► PharmacogenomicCleaner ─► geno_key column
                                                  │
                            DoubleTowerDataset ────┴─► GraphCache (lazy or preload)
                                                  │      ▲
                                                  │      └─ src/library/*.pt
                                                  │
                            DoubleTowerCollater ──────► PyG Batch ──► StandardTrainer
                                                                     │
                                                                     └─► src/pgen_model/
```

## API surface

| Endpoint                                | Purpose                                           |
| --------------------------------------- | ------------------------------------------------- |
| `GET /health`                           | Liveness — always 200 if process is up.           |
| `GET /ready`                            | Readiness — true once a model has been loaded.    |
| `GET /v1/models`                        | List available trained models with feature/target spec. |
| `GET /v1/models/{name}`                 | Full ModelConfig for one model.                   |
| `POST /v1/predict`                      | One (drug_cid, StarAllele) → PredictionResult.    |
| `POST /v1/predict/batch`                | ≤100 pairs in one call.                           |
| `GET /v1/library/drugs`                 | Paginated drug-graph catalog.                     |
| `GET /v1/library/genes`                 | Paginated gene-graph catalog.                     |
| `GET /v1/library/genes/{symbol}`        | Variants stored for one gene.                     |

Run with `uvicorn src.api.main:app --host 0.0.0.0 --port 8000`. Auto-OpenAPI at `/docs`.

## Conventions

- **Pydantic at every boundary.** New code accepting external input (HTTP, CLI args, TSV rows, TOML) goes through `src/domain/` or `src/config/` models — `dict[str, Any]` is banned in those layers.
- **Coordinates are 1-based.** `Position.pos` matches FASTA/VCF conventions. Anything 0-based is internal-only and named `_0based`.
- **Genome build is part of every Position.** Build mismatch is a fail-fast `BioinformaticsError`, not a warning. See `src/genomics/variant_val.py::iter_variants` for the pattern.
- **Star alleles come from `data/dicts/star_alleles.tsv`.** Don't reintroduce hardcoded tables in code. Add a row to the TSV instead.
- **No `shell=True`.** Use argv lists. For pipes, use `subprocess.Popen` plumbing (see `src/genomics/ngs_pipeline.py::map_reads`).
- **No `input()` in library code.** Interactive prompts live in `src/interface/cli.py`. Library functions take parameters or return generators.
- **English everywhere.** Spanish identifiers/comments in `src/` are tech debt to fix; `BACKUPS/` is exempt.
- **No emoji in log messages.** Emoji are for `ConsoleIO` user-facing output.
- **Schema dimensions are tested.** `tests/unit/data/test_library_drugs.py` and `test_library_genes.py` pin the 25/7 (drug) and 9/3 (gene) feature counts so accidental changes break CI before invalidating trained models.

## Outstanding tech debt

These are the remaining items from `Ref.md` that were intentionally deferred:

1. **Phase 9 — package rename.** Rename the package from `src` → `pharmagen` so users `from pharmagen.api import ...`. Touches every import; not done because the structure is still settling.
2. **CLI consolidation.** Move `src/interface/cli.py` and `src/interface/ui.py` to `src/cli/` once the menu workflows are split.
3. **`src/utils/io.py` shim removal.** Once nothing imports `DataLoaderUtils` from `src.utils.io`, delete the shim.
4. **`src/config/manager.py` shim removal.** Same idea — delete once `DIRS`/`SEED` callers all migrate to `get_settings()`.
5. **`src/genomics/vcf_handler/wrapper.py`.** Currently calls a non-existent C++ binary. Either remove or guard with `shutil.which("vcf_tool")`.
6. **`pyproject.toml` test config.** `[tool.pytest.ini_options].addopts` includes `--cov=pharmagen` which doesn't exist as a package; tests run with `--override-ini="addopts="`.
7. **Library schema versioning.** The existing `src/library/drugs/*.pt` files were built before the 25-feature schema was finalized; some artifacts have 24 features. The new builder produces the canonical 25/7 — old artifacts need rebuilding.
