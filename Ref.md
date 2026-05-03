# Pharmagen — `src/` Refactor Plan

> **Goal:** Stabilize the mid-refactor codebase, then incrementally restructure `src/` to follow Zen of Python, SOLID, modern bioinformatics conventions, and add a Pydantic-based domain layer + a FastAPI service for inference.
>
> **Context:** The repo is currently un-runnable. `src/pipeline.py` and the training engine import ten utility modules that were deleted from `src/utils/` during a prior refactor. Their content is preserved in `src/CAJON DE SASTRE/`. Two genomics modules import a non-existent `src.cfg.config`. Some files mix Spanish/English. Configs are dict-based with no validation.
>
> This plan is phased so each phase is independently committable, leaves the codebase in a runnable state, and can be paused for review.

---

## 0. Guiding Principles

Every change in this refactor must justify itself against these rules:

- **Zen of Python** — Explicit > implicit. Flat > nested. Readability counts. Errors should never pass silently. One obvious way to do it.
- **SOLID** — Each module owns one reason to change. Open for extension, closed for modification (factories/registries). Depend on abstractions, not concretions.
- **Bioinformatics conventions**
  - Coordinates are 1-based unless explicitly suffixed `_0based`.
  - Genome build is part of every `Variant` / `Position`; mismatched builds are an error, not a warning.
  - Star-allele naming follows `GENE*allele` with a single source of truth (no hardcoded tables in code).
  - VCF/FASTA I/O goes through `pysam`/`pyfaidx`; never hand-parse.
  - English is the project language. Translate Spanish identifiers/docstrings/comments.
- **Type-first** — Every public boundary is typed. Configs, requests, and bio entities are Pydantic v2 models, not dicts.
- **Fail fast** — Invalid input raises at the boundary. No silent fallbacks for missing files, missing alleles, or shape mismatches.
- **No magic at import time** — Modules expose factories and singletons through getters; they don't read TOMLs, create directories, or build lookup tables on import.

---

## 1. Target Layout (after the refactor)

```
src/
├── __init__.py                    # Package version, re-exports
├── pipeline.py                    # Thin entry point, delegates to pipelines/
├── api/                           # NEW — FastAPI service
│   ├── __init__.py
│   ├── main.py                    # FastAPI app factory + lifespan
│   ├── deps.py                    # DI: model loader, settings, predictor
│   ├── schemas.py                 # Request/response Pydantic models
│   └── routers/
│       ├── health.py              # GET /health, /ready
│       ├── models.py              # GET /v1/models
│       ├── predict.py             # POST /v1/predict (single + batch)
│       └── library.py             # GET /v1/library/{drugs,genes}
├── cli/                           # was src/interface/
│   ├── __init__.py
│   ├── app.py                     # main_menu_loop
│   ├── ui.py                      # ConsoleIO, Spinner, ProgressBar
│   ├── notices.py                 # was interface/io.py:print_gnu_notice / warranty / conditions
│   └── workflows/                 # one file per submenu
│       ├── train.py
│       ├── predict.py
│       └── genomics.py
├── config/
│   ├── __init__.py                # Public API: get_settings(), get_model_config()
│   ├── settings.py                # Pydantic Settings (env vars + paths)
│   ├── models.py                  # Pydantic ModelConfig + OptunaSpec
│   └── data/                      # The TOML files live here
│       ├── settings.toml
│       ├── paths.toml
│       └── models.toml
├── domain/                        # NEW — Pure Pydantic domain models
│   ├── __init__.py
│   ├── drug.py                    # Drug (Mol + SMILES + CID)
│   ├── variant.py                 # Variant, Genotype, GenomeBuild
│   ├── gene.py                    # Gene, StarAllele
│   ├── graph.py                   # GraphMetadata, GraphPair
│   └── prediction.py              # PredictionRequest, PredictionResult
├── data/
│   ├── __init__.py
│   ├── datasets.py                # PyTorch Dataset (slim)
│   ├── encoders.py                # PGenProcessor extracted
│   ├── cache.py                   # GraphCache (extracted from DoubleTowerDataset)
│   ├── collators.py
│   ├── loaders.py                 # CSV/TSV loaders (was interface/io.py:DataLoaderUtils)
│   ├── normalize.py               # MultiLabelNormalizer, Stratifier
│   └── library/
│       ├── __init__.py
│       ├── builder.py             # Orchestrator (was lib_builder_v2.py)
│       ├── drug_graphs.py         # SMILES → graph (RDKit)
│       └── gene_graphs.py         # VCF/variants → graph
├── genomics/
│   ├── __init__.py
│   ├── reference.py               # was ref_genome.py — GRCh38 manager
│   ├── pipelines.py               # was ngs_pipeline.py — NGS QC pipeline
│   ├── vcf.py                     # was variant_val.py — VCF parsing
│   └── star_alleles.py            # NEW — load star-allele table from data/dicts/
├── modeling/
│   ├── __init__.py
│   ├── architectures/
│   │   ├── __init__.py
│   │   ├── towers.py              # GATv2Tower
│   │   ├── two_tower.py           # PharmagenTwoTower
│   │   └── factory.py             # create_model (single, not duplicated)
│   ├── training/                  # was modeling/engine/
│   │   ├── __init__.py
│   │   ├── trainer.py             # StandardTrainer
│   │   ├── optuna_trainer.py      # OptunaTrialTrainer (separated)
│   │   ├── tuner.py
│   │   ├── losses.py              # restored from CAJON
│   │   ├── metrics.py             # restored from CAJON
│   │   ├── checkpoint.py          # restored from CAJON
│   │   └── factory.py             # OptimizerFactory, LossFactory
│   └── inference/
│       ├── __init__.py
│       └── predictor.py
├── pipelines/                     # NEW — high-level orchestration
│   ├── __init__.py
│   ├── train.py                   # train_pipeline (was src/pipeline.py)
│   ├── predict.py
│   └── library_build.py
└── utils/                         # General-purpose only
    ├── __init__.py
    ├── exceptions.py              # cleaned: rename PharmagenMemoryError, drop multi-inheritance hack
    ├── logging.py                 # was logger.py — fix Spanish comment, parameterize libs to silence
    ├── memory.py                  # restored from CAJON
    ├── system.py                  # restored from CAJON
    └── types.py                   # was pgen_types.py — TypeAliases
```

> **Key removals:** `src/types/` (folded into `src/domain/`), `src/interface/` (split into `src/cli/` + `src/api/`), `src/CAJON DE SASTRE/` (moved to `BACKUPS/`), `src/config/loader.py` (duplicate of manager.py).
>
> **Note on package naming:** `pyproject.toml`'s setuptools config (`packages.find` + `where = ["./*.py", "src"]`) is malformed. We will fix it in Phase 1 to `where = ["src"]`. A future Phase 9 (optional) renames the package from `src` to `pharmagen` for proper imports (`from pharmagen.api ...`); this is out of scope for the initial refactor to avoid touching every file.

---

## 2. Phasing

Each phase is a separate commit (or small stack of commits) and ends with the code in a runnable state. Phases 0–1 are non-negotiable stabilization. Phases 2–8 are open for redirection.

### Phase 0 — Triage & Backup *(target: ~30 min, blocks everything)*

1. Tag the current HEAD: `git tag pre-refactor-2026-05`.
2. Move `src/CAJON DE SASTRE/` → `BACKUPS/cajon_de_sastre_pre_refactor/`. The files there are the source of truth for restored utilities.
3. Verify `BACKUPS/` is gitignored (it already is — `BACKUPS/` line in `.gitignore`).
4. Skim `src/library/library_archive.tar.gz` and `src/pgen_model/` — confirm we don't need to touch them.

**Acceptance:** `src/` no longer contains "CAJON DE SASTRE". Existing artifacts preserved.

---

### Phase 1 — Stabilize Imports *(target: 4–6 hr, blocks Phases 2+)*

The codebase must import. We restore the deleted utilities **as-is** under `src/utils/` first (no redesign yet — that's Phase 4). Pure migration.

1. Restore from `BACKUPS/cajon_de_sastre_pre_refactor/` into `src/utils/`:
   - `checkpoint.py`, `losses.py`, `memory.py`, `metrics.py`, `module_builder.py`, `system.py`, `validation.py`, `data_utils.py`
2. Restore `src/utils/io.py` from `BACKUPS/.../io.py` if present, OR add a thin shim that re-exports `DataLoaderUtils` from `src/interface/io.py`. Decide based on which file is more recent.
3. Fix wrong import paths:
   - `src/genomics/ref_genome.py:21` — `src.cfg.config` → `src.config.manager`
   - `src/genomics/ngs_pipeline.py:17` — same
4. Fix `src/types/drugs.py` Pydantic config: add `model_config = ConfigDict(arbitrary_types_allowed=True)` so RDKit `Mol` and PyG `Data` are accepted.
5. Fix `pyproject.toml`'s setuptools `packages.find`:
   - Was: `where = ["./*.py", "src"]` (malformed glob)
   - To: `where = ["src"]`
6. Smoke test:
   ```bash
   python -c "from src.pipeline import train_pipeline; print('ok')"
   python -c "from src.modeling.engine.tuner import run_optuna_study; print('ok')"
   python -c "from src.modeling.engine.predictor import PGenPredictor; print('ok')"
   ```
7. Run existing pytest collection (no execution required — just `pytest --collect-only`) to confirm imports.

**Acceptance:** All three smoke imports succeed. `pytest --collect-only` runs without ImportError. CLI `python main.py --help` prints usage.

**Commit:** `fix(refactor): restore deleted utility modules and correct import paths`

---

### Phase 2 — Domain Models *(target: 6–8 hr)*

Build a Pydantic v2 `domain/` package as the type backbone. Nothing else is wired up yet — these are pure data classes with validators.

1. Create `src/domain/` with:
   - `drug.py` — `Drug` (move from `src/types/drugs.py`, add `ConfigDict(arbitrary_types_allowed=True)`, validate SMILES via RDKit, validate CID > 0).
   - `variant.py` — `GenomeBuild` enum (GRCh37, GRCh38), `Position` (chrom, pos, build), `Variant` (position + ref + alt + type), `Genotype` (variant + zygosity + sample_id). Position validation: 1-based, chrom matches FASTA naming.
   - `gene.py` — `Gene` (HGNC symbol, optional ENSG), `StarAllele` (gene + label like `*1`, `*4`, function: increased/normal/decreased/no-function).
   - `graph.py` — `GraphMetadata` (id, kind: drug/gene, source, num_nodes, num_edges, feature_dim, edge_dim).
   - `prediction.py` — `PredictionRequest` (drugs: list[CID], genotype: list[StarAllele]), `PredictionResult` (target → label + confidence).
2. Add field validators (e.g., chromosome normalization, SMILES sanity check, allele label regex).
3. Delete `src/types/` (its only inhabitant was `Drug`, now in `src/domain/drug.py`).
4. Add tests in `tests/unit/domain/`.

**Acceptance:** All domain models have ≥1 unit test each (valid + invalid case). `mypy src/domain/` is clean.

**Commit:** `refactor(domain): introduce Pydantic v2 domain models`

---

### Phase 3 — Configuration as Pydantic Settings *(target: 4–6 hr)*

Replace dict-based config with Pydantic-validated objects. `dict[str, Any]` is banned in `src/config/`.

1. Create `src/config/settings.py` using `pydantic-settings`:
   - `Settings` model: env vars (LOG_LEVEL, DATA_DIR, MODELS_DIR, …) with TOML defaults.
   - Loads `data/settings.toml` and `data/paths.toml` once, exposes `get_settings()` (lru_cached).
2. Create `src/config/models.py`:
   - `OptunaSpec` — discriminated union of `IntRange | FloatRange | LogRange | Categorical` parsed from TOML lists like `["log", 1e-4, 1e-2]`.
   - `ModelConfig` — features, targets, params, optuna, dimensions, etc.
   - `get_model_config(name) -> ModelConfig`.
3. Move TOMLs from `src/config/` → `src/config/data/`. Update loaders.
4. Delete `src/config/loader.py` (duplicate).
5. Move side effects: directory creation moves from import-time to a `Settings.ensure_dirs()` method, called explicitly from `main.py`.
6. Update all callers:
   - `from src.config.manager import DIRS, SEED, MULTI_LABEL_COLS, get_model_config`
     → `from src.config import get_settings, get_model_config`
   - Replace `cfg["features"]` etc. with `cfg.features` (attribute access).
7. Tests for config loading, OptunaSpec parsing, and validation errors.

**Acceptance:** No `dict[str, Any]` in `src/config/`. `get_model_config("TwoTowerGAT")` returns a typed `ModelConfig`. Test coverage ≥80% for `src/config/`.

**Commit:** `refactor(config): replace dict-based config with Pydantic Settings + ModelConfig`

---

### Phase 4.5 — Library Builder Refactor *(target: 6–10 hr)*

The library builder is critical infrastructure: it pre-computes drug + variant graphs offline so training can lazy-load from disk (the user's solution to limited compute). Two files exist today and only one mostly works:

- `src/data/lib_builder_polars.py` (883 LOC) — the working Polars/pyfaidx/RDKit/networkx implementation, but coupled by module globals (`GLOBAL_GENOME`, `GLOBAL_CHROM_MAPPING`), hardcoded paths, Spanish comments, shell-script-based file organization, no resume support.
- `src/data/lib_builder_v2.py` (62 LOC) — abandoned Pydantic refactor with `...` stubs. Delete.

Replace with a clean `src/data/library/` package:

```
src/data/library/
├── __init__.py
├── chromosome.py     # CHROM ↔ RefSeq accession map
├── config.py         # LibraryBuildConfig (Pydantic) + path resolution
├── drugs.py          # smiles_to_graph + DrugGraphBuilder; CID-keyed .pt files
├── genes.py          # GenomicGraphBuilder + variant validation against FASTA
├── pgx.py            # Per-gene PharmVar VCF folder loader
├── manifest.py       # Resume tracking — JSON manifest of completed work
├── organize.py       # Pure-Python file organization (no bash/PowerShell)
├── builder.py        # Top-level orchestrator
└── __main__.py       # CLI: python -m src.data.library
```

**Invariants to preserve** (consumers in `src/data/graph_indexing.py`):
- Drug filename: `<cid>_<safe_name>.pt`
- Gene filename: `<gene>_<variant>.pt` in `<gene>/` subdirs
- "star" prefix in variant name → `*` (e.g., `star4` → `*4`) on the consumer side
- 25 drug node features, 7 drug edge features
- 9 gene node features, 3 gene edge features

**Key fixes** beyond the pure rewrite:
- Eliminate module-level globals; FASTA handle is a constructor arg.
- Use `get_settings().paths` instead of `BASE_DIR = Path("data")` etc.
- Use `data/dicts/star_alleles.tsv` for known function metadata where possible.
- Skip on existing `.pt` (resume); `--force` to overwrite.
- Replace shell organize scripts with `pathlib.Path.rename` / `shutil.move`.
- `logging` instead of `print`.
- `BioinformaticsError`/`DataError` from `src.utils.exceptions`.

**Acceptance:** old `.pt` artifacts in `src/library/` remain valid (consumer-side `GraphIndexBuilder` still works). `python -m src.data.library --help` runs. Smoke tests for SMILES→graph dimension stability and `safe_filename`.

**Commit:** `refactor(library): rewrite library builder as src/data/library/ package`

---

### Phase 4 — Decompose God Objects *(target: 12–16 hr)*

Targets: `DoubleTowerDataset`, `DataLoaderUtils`, `PGenTrainer`. Each is split along SRP lines.

1. **`DoubleTowerDataset`** (~300 lines) → split into:
   - `data/cache.py::GraphCache` — RAM caching, eviction, prefetch.
   - `data/encoders.py::EncoderRegistry` — fit/transform/save/load sklearn encoders.
   - `data/datasets.py::DoubleTowerDataset` — slim Dataset that composes the above.
2. **`DataLoaderUtils`** (~210 lines, currently in `src/interface/io.py`) → split into:
   - `data/loaders.py::TabularLoader` — read CSV/TSV with schema validation.
   - `data/normalize.py::MultiLabelNormalizer`, `data/normalize.py::Stratifier`.
   - `genomics/star_alleles.py::StarAlleleMap` — loads PharmGKB-style table from `data/dicts/`.
3. **`PGenTrainer`** → two trainers sharing a common `TrainingLoop` abstract base:
   - `modeling/training/trainer.py::StandardTrainer` — checkpointing, full logging.
   - `modeling/training/optuna_trainer.py::OptunaTrialTrainer` — minimal logging, no checkpointing.
4. Hoist hardcoded star-allele table out of `interface/io.py` into a CSV/TSV in `data/dicts/star_alleles.tsv` (still gitignored, but with a sample committed).
5. Tests: each new class gets a focused unit test.

**Acceptance:** Each new module is <250 LOC. Old import paths still work via shims for one phase. Existing tests pass.

**Commit (or split into 3):** `refactor(data): decompose god objects` etc.

---

### Phase 5 — Bioinformatics Hardening *(target: 8–10 hr)*

1. `src/genomics/vcf.py` (was `variant_val.py`):
   - Replace partial `decodificar_genotipo` with a full parser using `pysam.VariantFile`.
   - Yields `Genotype` (Phase 2 domain) objects; raises `BioinformaticsError` on malformed VCFs.
   - Validates genome build matches `Settings.genome_build`.
   - Removes interactive `input()` from `seleccionar_vcf` — separate the CLI prompt (in `cli/workflows/genomics.py`) from the library function.
2. `src/genomics/reference.py` (was `ref_genome.py`):
   - Resumable download (HTTP Range), checksum verification (md5 from Ensembl).
   - Background download via `Settings.background_download` flag.
3. `src/genomics/pipelines.py` (was `ngs_pipeline.py`):
   - Replace `shell=True` subprocess calls with `subprocess.run([...], check=True)`.
   - Add timeouts and structured logging.
4. `src/genomics/star_alleles.py`:
   - Loads `data/dicts/star_alleles.tsv` (gene, allele, function, defining_variants).
   - Validates rows with Pydantic `StarAllele` model.
5. `src/genomics/vcf_handler/wrapper.py`:
   - Currently calls a non-existent C++ binary. Either remove (preferred) or guard behind a `if shutil.which("vcf_tool")` check that raises a clear error.

**Acceptance:** No `input()` calls in `src/genomics/`. No `shell=True`. VCF parsing works on a known sample. Star alleles loaded from data, not code.

**Commit:** `refactor(genomics): adopt bio conventions and remove interactive coupling`

---

### Phase 6 — English-Only & Cleanup *(target: 6–10 hr)*

1. Audit all `src/` for Spanish — translate identifiers, comments, docstrings:
   - `decodificar_genotipo` → `decode_genotype`
   - `seleccionar_vcf` → `select_vcf`
   - `descarga_del_genoma` → `download_genome`
   - Comments like `# ya configurado`, `# Normalizaciones de grafo`, etc.
2. Move all license headers into a single `LICENSE_HEADER.txt` and apply via `ruff` rule `CPY` or a one-off pre-commit hook (don't repeat the 17-line header in every file).
3. Standardize logging: emoji removed from log messages; emoji are for `ConsoleIO` user-facing output only.
4. Remove unused imports (e.g., `GraphValidator` in `data/datasets.py:25` — no usages).
5. Resolve the `MemoryError` shadowing: rename to `PharmagenMemoryError` in `src/utils/exceptions.py` and update imports.
6. Drop `ValidationError`'s multi-inheritance hack (`PharmagenException, IndexError, ValueError`) — pick one parent.

**Acceptance:** `rg -n '[áéíóúñ]|[A-Z][a-zA-Záéíóú_]*[áéíóúñ]' src/` returns zero hits. `ruff check src/` clean.

**Commit:** `refactor: localize project to English and clean up exception hierarchy`

---

### Phase 7 — FastAPI Service *(target: 10–14 hr)*

A read-only inference API. Training stays CLI-only for now.

1. `src/api/main.py`:
   - `create_app()` factory pattern.
   - `lifespan` context loads `Settings` and warm-loads the default model via `PGenPredictor`.
2. `src/api/deps.py`:
   - DI for `Settings`, `PGenPredictor`, request-scoped logging.
3. `src/api/schemas.py`:
   - `PredictRequest` — drugs (list of CIDs or names), genotype (list of star alleles).
   - `PredictResponse` — phenotype_category + per-target probabilities + model version.
   - `ModelInfo`, `LibraryEntry`.
4. Routers:
   - `health.py` — `GET /health` (liveness), `GET /ready` (model loaded).
   - `models.py` — `GET /v1/models` (list), `GET /v1/models/{name}` (config + metrics).
   - `predict.py` — `POST /v1/predict` (single), `POST /v1/predict/batch` (≤100 items per request).
   - `library.py` — `GET /v1/library/drugs` (paginated), `GET /v1/library/genes`.
5. Add `uvicorn` to dependencies if not present (it's pulled in by FastAPI but better explicit).
6. Run with: `uvicorn src.api.main:app --reload`.
7. Auto-generated OpenAPI at `/docs`.
8. Tests with `httpx.AsyncClient` against the app.

**Acceptance:** `curl localhost:8000/health` returns 200. `POST /v1/predict` returns a typed response. OpenAPI schema includes all endpoints.

**Commit:** `feat(api): add FastAPI inference service`

---

### Phase 8 — Tests, CI, Docs *(ongoing, ~ continuous)*

1. Coverage target ≥70% across `src/`.
2. CI: GitHub Actions workflow running `ruff check`, `mypy src`, `pytest`.
3. Update `CLAUDE.md` to reflect the new layout.
4. Update `README.md` with FastAPI usage.
5. Add `docs/ARCHITECTURE.md` describing the new module boundaries.

---

### Phase 9 (Optional, deferred) — Package Rename

Rename the package from `src` to `pharmagen` so users `from pharmagen.api import ...`. This touches every import in the project and isn't worth doing until the structure is stable. Track as a follow-up.

---

## 3. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Breaking the training pipeline mid-refactor | Each phase ends with a green smoke test and pytest collection. Phase 1 specifically restores **runnability** before any redesign. |
| Trained model artifacts (`.pt` files) become incompatible with refactored code | Predictor loads checkpoints with explicit version checks. Old artifacts continue to work by reading `MODEL_VERSION` from the checkpoint dict. |
| Touching too many files at once → unreviewable diff | Phases are independently committable. No phase exceeds ~1500 lines changed. |
| Pydantic v2 strict mode rejects RDKit `Mol` / PyG `Data` | Use `model_config = ConfigDict(arbitrary_types_allowed=True)` on relevant domain models (Drug, GraphMetadata). |
| Spanish translation introduces typos in identifiers exposed in checkpoints | Keep encoder/checkpoint **keys** stable (e.g., `"phenotype_category"` stays). Only rename *internal* identifiers. |
| FastAPI + heavy model load → slow cold start | Use `lifespan` to load model once at startup. Add `--workers 1 --preload` guidance. |

---

## 4. Out of Scope

These came up during analysis but won't be tackled here without explicit ask:

- Migrating from `polars` to `pandas` or vice versa (codebase uses both — fine for now).
- Replacing PyTorch Geometric with DGL or another graph lib.
- Distributed training (multi-GPU / DDP).
- Model serving with `torchscript` / `onnx`.
- Async background job queue for batch predictions (Celery/Arq) — Phase 7 keeps it sync.
- Migrating `setuptools` → `hatchling`/`uv build` (mentioned but separate concern).

---

## 5. Execution Order in This Session

1. **Now:** Write this `Ref.md` (done).
2. **Now:** Phase 0 (backup `CAJON DE SASTRE`).
3. **Now:** Phase 1 (restore utilities + fix imports + smoke test).
4. **Pause for review:** Get sign-off on the new layout (Section 1) and any redirection on Phases 2–7 before starting Phase 2. Domain modeling and config restructuring are opinionated changes you'll want to steer.

After your review, phases run sequentially. Estimated total effort: **8–12 working days** of focused work, give or take. We won't finish in one session, but you'll have a runnable codebase and a clear roadmap before we stop.
