# Pharmagen Architecture

This document describes the runtime layout of `src/` after the 2026-Q2 refactor
(see [`Ref.md`](../Ref.md) for the phased history). It is the canonical
reference for new contributors and for AI assistants working on the codebase.

## High-level picture

Pharmagen exposes four user-facing surfaces — an interactive CLI, a headless
CLI, a FastAPI service, and a programmatic Python API. All of them sit on top
of the same domain models, configuration, and trained model artifacts.

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
│ src/        │  │ src/api/     │  │ src/data/  │  │ src/model/training/      │
│ interface/  │  │ (FastAPI     │  │ library/   │  │ (StandardTrainer +       │
│ (interactive│  │  inference   │  │ (offline   │  │  OptunaTrialTrainer over │
│  menu, IO)  │  │  service)    │  │  graph     │  │  a TrainingLoop ABC)     │
│             │  │              │  │  builder)  │  │                          │
└──────┬──────┘  └──────┬───────┘  └────┬───────┘  └─────────────┬────────────┘
       │                │               │                        │
       │                ▼               ▼                        ▼
       │     ┌──────────────────────────────────────┐   ┌────────────────┐
       │     │ src/data/                            │   │ src/library/   │
       │     │   datasets.py (DoubleTowerDataset)   │   │ drugs/*.pt     │
       │     │   cache.py    (GraphCache)           │   │ gene_graphs/   │
       │     │   encoders.py (TargetEncoder)        │   │   <gene>/*.pt  │
       │     │   loaders.py  (TabularLoader)        │   └───────▲────────┘
       │     │   cleaning.py (PharmacogenomicCleaner)        │
       │     │   normalize.py · collator.py · graph_indexing.py
       │     └──────────────────────────────────────┘           │
       │                                                        │
       └─► src/pipeline.py orchestrates training; predictions read this on-disk library.
```

## Module map

```
src/
├── pipeline.py             Training orchestrator (train_pipeline)
│
├── api/                    FastAPI inference service
│   ├── main.py             create_app() factory + lifespan
│   ├── deps.py             PredictorRegistry + DI helpers
│   ├── schemas.py          HTTP request / response envelopes
│   └── routers/            health · models · predict · library
│
├── config/                 Pydantic Settings
│   ├── __init__.py         Public API: get_settings, get_model_config, …
│   ├── settings.py         Settings (env-overridable via PHARMAGEN_*)
│   ├── paths.py            Paths model with explicit ensure_dirs()
│   ├── models.py           ModelConfig + OptunaSpec discriminated union
│   └── data/               TOML data files (settings.toml, paths.toml, models.toml)
│
├── core/                   Cross-cutting primitives
│   ├── exceptions.py       PharmagenException hierarchy
│   ├── log.py              setup_logging
│   └── validation.py       ConfigValidator + DataValidator
│
├── data/                   Data loading + preprocessing
│   ├── datasets.py         DoubleTowerDataset (composes cache + encoder)
│   ├── cache.py            GraphCache + GraphDims + make_empty_graph
│   ├── encoders.py         TargetEncoder (single + multi-label)
│   ├── loaders.py          TabularLoader (CSV / TSV with project schema)
│   ├── cleaning.py         GenoKeyBuilder + PharmacogenomicCleaner
│   ├── normalize.py        MultiLabelNormalizer + Stratifier
│   ├── collator.py         DoubleTowerCollater (PyG batching)
│   ├── graph_indexing.py   GraphIndexBuilder (walks src/library/)
│   └── library/            Offline graph builder
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
├── domain/                 Pydantic v2 domain models
│   ├── drug.py             Drug + Drug.from_smiles()
│   ├── variant.py          GenomeBuild · Position · Variant · Genotype
│   ├── gene.py             Gene · StarAllele · AlleleFunction
│   ├── graph.py            GraphKind · GraphMetadata · GraphPair
│   └── prediction.py       PredictionRequest · PredictionResult
│
├── genomics/               Bioinformatics integrations
│   ├── ref_genome.py       GRCh38 download + samtools / bwa indexing
│   ├── ngs_pipeline.py     FastQC → BWA → Picard → Freebayes → VEP
│   ├── variant_val.py      iter_variants() with build-mismatch detection
│   └── star_alleles.py     StarAlleleMap (loads data/dicts/star_alleles.tsv)
│
├── interface/              Console UI + interactive CLI
│   ├── cli.py              main_menu_loop
│   ├── ui.py               ConsoleIO · Spinner · ProgressBar
│   └── io.py               JSON helpers + DataLoaderUtils facade
│
├── library/                On-disk graph cache (built by src/data/library)
│   ├── drugs/              <cid>_<name>.pt
│   ├── gene_graphs/        <gene>/<gene>_<variant>.pt
│   └── build_manifest.json Resume manifest (atomic JSON)
│
└── model/                  Model + training
    ├── architectures/      Towers + factory
    │   ├── gnn.py          GATv2Tower · PharmagenTwoTower
    │   └── layers.py       create_gnn_model
    ├── training/           Trainer split
    │   ├── loop.py         TrainingLoop ABC (shared concerns)
    │   ├── standard.py     StandardTrainer (compile + checkpoints + tqdm)
    │   └── optuna_trainer.py  OptunaTrialTrainer (trial reporting + pruning)
    ├── engine/             Inference + hyperparameter search
    │   ├── predictor.py    PGenPredictor (loads weights + encoders)
    │   └── tuner.py        run_optuna_study (study orchestration)
    ├── checkpoint.py       CheckpointManager
    ├── factories.py        LossFactory + OptimizerFactory
    └── losses.py           MultiTaskUncertaintyLoss + focal / asymmetric
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

The legacy `src.config.manager` shim was removed during post-refactor cleanup.
**New code imports from `src.config` directly.**

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

| Endpoint                                | Purpose                                                  |
| --------------------------------------- | -------------------------------------------------------- |
| `GET /health`                           | Liveness — always 200 if the process is up.              |
| `GET /ready`                            | Readiness — `true` once a model has been loaded.         |
| `GET /v1/models`                        | List available trained models with feature/target spec.  |
| `GET /v1/models/{name}`                 | Full `ModelConfig` for one model.                        |
| `POST /v1/predict`                      | One `(drug_cid, StarAllele)` → `PredictionResult`.       |
| `POST /v1/predict/batch`                | Up to 100 pairs per call.                                |
| `GET /v1/library/drugs`                 | Paginated drug-graph catalog.                            |
| `GET /v1/library/genes`                 | Paginated gene-graph catalog.                            |
| `GET /v1/library/genes/{symbol}`        | Variants stored for one gene.                            |

Run with `uvicorn src.api.main:app --host 0.0.0.0 --port 8000`. Auto-generated
OpenAPI docs at `/docs`.

## Conventions

- **Pydantic at every boundary.** New code accepting external input (HTTP, CLI
  args, TSV rows, TOML) goes through `src/domain/` or `src/config/` models —
  `dict[str, Any]` is banned in those layers.
- **Coordinates are 1-based.** `Position.pos` matches FASTA / VCF conventions.
  Anything 0-based is internal only and named `_0based`.
- **Genome build is part of every Position.** Build mismatch is a fail-fast
  `BioinformaticsError`, not a warning. See
  [`src/genomics/variant_val.py::iter_variants`](../src/genomics/variant_val.py)
  for the pattern.
- **Star alleles come from `data/dicts/star_alleles.tsv`.** Do not reintroduce
  hardcoded tables in code. Add a row to the TSV instead.
- **No `shell=True` in subprocess calls.** Use argv lists. For pipes, use
  `subprocess.Popen` plumbing (see
  [`src/genomics/ngs_pipeline.py`](../src/genomics/ngs_pipeline.py)).
- **No `input()` in library code.** Interactive prompts live in
  [`src/interface/cli.py`](../src/interface/cli.py). Library functions take
  parameters or return generators.
- **English everywhere in `src/`.** Spanish identifiers / comments are tech
  debt to clear up; `BACKUPS/` is exempt.
- **No emoji in log messages.** Emoji are for `ConsoleIO` user-facing output
  only. Log messages use the form
  `logger.info("Doing X (sample=%s)", sample_id)`.
- **Schema dimensions are tested.**
  `tests/unit/data/test_library_drugs.py` and `test_library_genes.py` pin the
  25/7 (drug) and 9/3 (gene) feature counts so accidental changes break CI
  before silently invalidating trained models.
- **`src/core/` for cross-cutting code.** Import via
  `from src.core import EncoderError`, `from src.core import setup_logging`,
  etc. The old flat `src.utils` namespace has been removed.

## Outstanding tech debt

These items were intentionally deferred during the refactor (see
[`Ref.md`](../Ref.md) and [`CLAUDE.md`](../CLAUDE.md)):

1. **Phase 8 — CI workflow and final docs sweep.** No
   `.github/workflows/` yet.
2. **Package rename.** Rename `src` → `pharmagen` so users
   `from pharmagen.api import ...`. Touches every import; deferred while
   the structure is still settling.
3. **Library artefact relocation.** `src/library/` mixes generated `.pt`
   caches with an `__init__.py`. A future phase should move the artefacts
   to `data/library/` and turn `src/library/` into pure code (or remove
   it). `src/data/datasets.py` still hard-codes `PROJECT_ROOT / "src" /
   "library"`; once the artefacts move, switch to
   `get_settings().paths.<…>`.
4. **`src/model/engine/` shares too little with `src/model/training/`.**
   `predictor.py` and `tuner.py` were not restructured during the trainer
   split and could share a base with `TrainingLoop`.
5. **`main.py` (~300 LOC) mixes CLI parsing, logging setup, and dispatch.**
   A future `src/cli/app.py` would let `main.py` become a one-liner.
6. **`tests/integration/` is empty.** Needs smoke tests (pipeline import,
   predict round-trip).
7. **Library schema migration.** Some existing `src/library/drugs/*.pt`
   artefacts were built before the 25-feature schema was finalized and need
   rebuilding via `python -m src.data.library --force`.
