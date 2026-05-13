# Optimization Summary

A cross-cutting summary of the engineering decisions baked into the
post-refactor codebase: how memory is managed, how the code is structured
along SOLID lines, and how errors surface. For the phased rewrite that got us
here, see [`Ref.md`](../Ref.md); for runtime layout, see
[`docs/ARCHITECTURE.md`](ARCHITECTURE.md).

## 1. Memory management

Pharmagen is designed to run on commodity hardware. The pipeline picks safe
defaults automatically; the operator overrides only when they need to.

| Concern                       | Where it lives                              |
| ----------------------------- | ------------------------------------------- |
| Custom OOM exception          | `src/core/exceptions.py::PharmagenMemoryError` |
| Memory-aware preloading       | `src/pipeline.py` (`PRELOAD_THRESHOLD` rows) |
| Per-trial cleanup             | `src/model/training/optuna_trainer.py`       |
| Periodic GC during training   | `src/model/training/standard.py`             |

Operational guidance: [`docs/MEMORY_OPTIMIZATION.md`](MEMORY_OPTIMIZATION.md).

## 2. SOLID applied to the codebase

### Single Responsibility

The original `DoubleTowerDataset`, `PGenTrainer`, and `DataLoaderUtils` god-
objects were split into focused units:

- `src/data/datasets.py` — slim dataset that composes the cache + encoder.
- `src/data/cache.py` — `GraphCache` + `GraphDims`; the only owner of on-disk
  `.pt` files.
- `src/data/encoders.py` — `TargetEncoder` (single + multi-label).
- `src/data/loaders.py` — `TabularLoader` (CSV / TSV with project schema).
- `src/data/cleaning.py` — `GenoKeyBuilder` + `PharmacogenomicCleaner`.
- `src/data/normalize.py` — `MultiLabelNormalizer` + `Stratifier`.
- `src/model/training/loop.py` — `TrainingLoop` ABC.
- `src/model/training/standard.py` — `StandardTrainer`.
- `src/model/training/optuna_trainer.py` — `OptunaTrialTrainer` with trial
  reporting and pruning.

### Open/Closed

Factories let new components register without touching the consumer:

```python
from src.model.factories import LossFactory

LossFactory.register("my_loss", MyLossClass)
```

Both `LossFactory` and `OptimizerFactory` live in `src/model/factories.py`.

### Liskov Substitution

`StandardTrainer` and `OptunaTrialTrainer` both inherit from `TrainingLoop`
(`src/model/training/loop.py`) and return the same metric dict shape, so any
consumer can substitute one for the other.

### Interface Segregation

Domain models live in `src/domain/` and are imported piecewise. A caller that
only needs `StarAllele` imports only that — there is no monolithic
`pharmagen_types` module.

### Dependency Inversion

`StandardTrainer` accepts a `nn.Module`, a `torch.optim.Optimizer`, and any
`nn.Module` for loss. The concrete classes are wired in `src/pipeline.py`,
not inside the trainer.

## 3. Error model

A single exception hierarchy under `src/core/exceptions.py`:

```
PharmagenException
├── ConfigurationError       # invalid TOML / settings
├── DataError                # bad inputs to the pipeline
├── ModelError               # model creation / loading failures
├── EncoderError             # target encoder issues
├── GraphError               # PyG graph integrity
├── PharmagenMemoryError     # OOM with actionable context
├── HardwareError            # missing GPU, CUDA mismatch, …
├── BioinformaticsError      # genome-build mismatch, FASTA issues
├── ValidationError          # Pydantic-bridge violations
├── OptimizationError        # Optuna trial failures
├── TrainingError            # training-loop failures
├── ConvergenceError         # training failed to converge
└── ResourceError            # generic resource-availability problem
```

Import via `from src.core import ConfigurationError, DataError` — flat
imports from `src.utils` have been removed.

## 4. Validation framework

`src/core/validation.py` exposes:

- `ConfigValidator` — sanity-checks `ModelConfig` against the available
  feature / target columns and the `OptunaSpec` union.
- `DataValidator` — flags missing columns, NaNs, class imbalance, and dataset
  size before training begins (`MIN_DATASET_SIZE = 100`).

Both are invoked at the top of `src/pipeline.train_pipeline`, so failures land
early with actionable error messages instead of mysterious shape errors mid-
epoch.

## 5. Pydantic at every boundary

External input is parsed into Pydantic v2 models:

- **HTTP** — `src/api/schemas.py` envelopes wrap `PredictionRequest` /
  `PredictionResult` from `src/domain/`.
- **CLI / TOML** — `src/config/` resolves `Settings`, `Paths`, and
  `ModelConfig`.
- **Tabular** — `src/data/loaders.TabularLoader` rejects rows that do not
  match the project schema.
- **Bioinformatics** — `src/domain/variant.py` (Position, Variant, Genotype)
  rejects build mismatches up front; `src/genomics/variant_val.iter_variants`
  enforces FASTA agreement.

`dict[str, Any]` is banned in those layers. Adding a feature means adding a
field to a Pydantic model.

## 6. Subprocess safety

The NGS pipeline (`src/genomics/ngs_pipeline.py`) uses argv lists with no
`shell=True`. Pipes are stitched together with `subprocess.Popen`
(`map_reads`), which keeps shell-injection vectors closed even when the
caller controls a path or filename.

## 7. Documentation surface

| Document                             | Audience                                  |
| ------------------------------------ | ----------------------------------------- |
| [`README.md`](../README.md) / [`README_ESP.md`](../README_ESP.md) | First-time visitors |
| [`docs/QUICK_START.md`](QUICK_START.md) | Anyone running the tool                |
| [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) | New contributors                       |
| [`docs/LIBRARY_BUILDER.md`](LIBRARY_BUILDER.md) | Operators building the graph cache |
| [`docs/MEMORY_OPTIMIZATION.md`](MEMORY_OPTIMIZATION.md) | Training under tight budgets |
| [`docs/CODE_QUALITY.md`](CODE_QUALITY.md) | Patch authors                          |
| [`Ref.md`](../Ref.md)                | Maintainers, refactor history             |
| [`CLAUDE.md`](../CLAUDE.md)          | AI assistants                             |

## 8. What still needs doing

Tracked in [`Ref.md`](../Ref.md) and [`CLAUDE.md`](../CLAUDE.md):

- Phase 8 — CI workflow under `.github/workflows/` and a final docs sweep.
- Package rename `src/` → `pharmagen/`.
- Library artefact relocation (`src/library/` → `data/library/`).
- A shared base between `src/model/engine/{predictor,tuner}.py` and
  `src/model/training/loop.py`.
- Smoke-tests under `tests/integration/`.
- Schema migration for `.pt` artefacts predating the 25-feature drug schema.
