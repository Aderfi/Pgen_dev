# Quick Start

A guided tour through the four user-facing surfaces. For the full module map see `docs/ARCHITECTURE.md`; for the refactor history see `Ref.md`.

## 1. Install

```bash
git clone https://github.com/Aderfi/Pharmagen.git
cd Pharmagen
uv sync --extra dev               # creates .venv and installs everything
source .venv/bin/activate
```

Pharmagen is pinned to **Python 3.14** (see `.python-version`). It is managed with **uv** — `uv.lock` is the source of truth.

## 2. Build the offline graph library

The training pipeline lazy-loads pre-built drug + variant graphs from `src/library/`. Build them once:

```bash
python -m src.data.library                # full build (drugs + genes)
python -m src.data.library --only-gene CYP2D6 --skip-drugs  # quick verify
python -m src.data.library --force        # force-rebuild after schema changes
```

Inputs live under `data/`: `snp_data_output.tsv`, `drugs_cid.tsv`, `ref_genome/HSapiens_GChr38.fa`, and `haplotype_variants/<gene>/*.vcf`. See `docs/LIBRARY_BUILDER.md` for the input schemas and CLI options.

The build is **resumable** — interrupted runs pick up where they left off via `src/library/build_manifest.json`.

## 3. Train a model

### Standard training (CLI)

```bash
python main.py --mode train --model TwoTowerGAT \
    --input train_data/train_data.tsv --epochs 100
```

### Standard training (Python)

```python
from src.pipeline import train_pipeline

train_pipeline(
    model_name="TwoTowerGAT",
    csv_path="train_data/train_data.tsv",
    epochs=50,
    batch_size=32,
)
```

### Optuna hyperparameter search

```bash
python main.py --mode train --optuna --optuna-trials 50 --optuna-epochs 30
```

```python
from src.modeling.engine.tuner import run_optuna_study

run_optuna_study(
    model_name="TwoTowerGAT",
    csv_path="train_data/train_data.tsv",
    n_trials=50,
)
```

## 4. Inference (FastAPI)

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
# OpenAPI docs at http://localhost:8000/docs
```

Single prediction:

```bash
curl -X POST http://localhost:8000/v1/predict \
  -H 'Content-Type: application/json' \
  -d '{
        "drug_cid": 2244,
        "allele": {"gene": {"symbol": "CYP2D6"}, "allele": "4"}
      }'
```

Programmatic equivalent:

```python
from fastapi.testclient import TestClient
from src.api.main import create_app

client = TestClient(create_app())
resp = client.post("/v1/predict", json={
    "drug_cid": 2244,
    "allele": {"gene": {"symbol": "CYP2D6"}, "allele": "4"},
})
print(resp.json())
```

The API loads the model lazily on the first `/v1/predict` request — `/health` returns 200 even before any model is loaded. If trained artifacts are missing you'll get a clean `503` with the missing-file path in the `detail`.

Available endpoints (full list in `docs/ARCHITECTURE.md`):

| Method | Path                              | Purpose                                  |
| ------ | --------------------------------- | ---------------------------------------- |
| `GET`  | `/health`                         | Liveness probe.                          |
| `GET`  | `/ready`                          | Has a model been loaded?                 |
| `GET`  | `/v1/models`                      | List trained models.                     |
| `GET`  | `/v1/models/{name}`               | Full ModelConfig for one model.          |
| `POST` | `/v1/predict`                     | Single (drug, allele) prediction.        |
| `POST` | `/v1/predict/batch`               | ≤100 pairs in one call.                  |
| `GET`  | `/v1/library/drugs`               | Paginated drug-graph catalog.            |
| `GET`  | `/v1/library/genes`               | Paginated gene-graph catalog.            |
| `GET`  | `/v1/library/genes/{symbol}`      | Variants stored for one gene.            |

## 5. Inference (CLI menu)

```bash
python main.py            # interactive menu
```

Menu options:

1. Genomic Processing (ETL — VCF in, predictions out; not yet wired up)
2. Train Models
3. Predict (interactive single + file batch)
4. Advanced Analysis (placeholder)
5. Exit

## 6. Configuration

The settings system is `pydantic-settings`-based. Inspect resolved settings:

```python
from src.config import get_settings

settings = get_settings()
print(settings.project_name, settings.version)
print(settings.paths.data, settings.paths.models)
print(settings.multi_label_cols)
```

Override via environment variables prefixed `PHARMAGEN_`:

```bash
PHARMAGEN_LOG_LEVEL=DEBUG python main.py --mode train --model TwoTowerGAT
```

Per-model config:

```python
from src.config import get_available_models, get_model_config

print(get_available_models())                       # ['TwoTowerGAT']
cfg = get_model_config("TwoTowerGAT")
print(cfg.features, cfg.targets)
print(cfg.params)                                   # fixed hyperparameters
print(cfg.optuna["learning_rate"])                  # LogSpec(low=2e-4, high=8e-4)
```

The TOML files those values come from live in `src/config/data/`.

## 7. Domain models — the canonical types

Every public boundary speaks Pydantic. Examples:

```python
from src.domain import (
    Drug, Variant, Position, Genotype, Zygosity,
    Gene, StarAllele, AlleleFunction, GenomeBuild,
    PredictionRequest,
)

# Drug from SMILES (RDKit-validated)
drug = Drug.from_smiles(name="aspirin", cid=2244, smiles="CC(=O)Oc1ccccc1C(=O)O")

# Variant on chr17 (chromosome label normalized internally)
v = Variant(position=Position(chrom="chr17", pos=43_044_295), ref="G", alt="A")
print(v.variant_type)                                # VariantType.SNP

# Star allele resolved from the catalog at data/dicts/star_alleles.tsv
sa = StarAllele.parse("CYP2D6*4")
print(sa.label, sa.function.value)                   # CYP2D6*4 unknown

# Inference request (validated; extra='forbid')
req = PredictionRequest(drugs=[2244], genotype=[sa])
```

## 8. Testing

```bash
pytest tests/unit/ -q --override-ini="addopts="     # 237 tests, ~3 seconds
pytest tests/unit/api/                              # FastAPI tests
pytest tests/unit/data/test_library_*.py            # library-builder schema tests
```

The `--override-ini="addopts="` flag bypasses the stale `--cov=pharmagen` flag in `pyproject.toml` (Phase 8 cleanup pending).

## 9. Common gotchas

* **`pysam` not installed** — only needed for `src.genomics.variant_val`; safe to skip on Windows or in lightweight envs.
* **Aspirin graph has 24 features instead of 25** — the existing `src/library/drugs/*.pt` artifacts were built before the 25-feature schema was finalized. Rebuild with `python -m src.data.library --force` to get the canonical 25/7 schema.
* **`OutOfMemory` during Optuna** — drop `--optuna-trials` and `batch_size`; the tuner uses `preload_ram=False` automatically but VRAM still budgets per trial.
* **CLI menu uses Spanish strings somewhere** — that's tech debt slated for Phase 9. PR welcome.

## See also

- `docs/ARCHITECTURE.md` — the full module map.
- `docs/LIBRARY_BUILDER.md` — deep dive on the offline graph builder.
- `docs/CODE_QUALITY.md` — Zen of Python + SOLID examples.
- `docs/MEMORY_OPTIMIZATION.md` — OOM avoidance during training.
- `Ref.md` — the phased refactor plan with progress.
