# Pharmagen

> **Pharmacogenetic prediction and therapeutic efficacy via deep learning.**

![Python](https://img.shields.io/badge/python-3.14-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![PyG](https://img.shields.io/badge/PyTorch%20Geometric-2.5+-3c3c3c.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.135+-009688.svg)
![Pydantic](https://img.shields.io/badge/Pydantic-v2-e92063.svg)
![License](https://img.shields.io/badge/license-GPLv3-blue.svg)
![Status](https://img.shields.io/badge/status-Beta-orange.svg)

Pharmagen maps a patient's genomic profile (gene / star allele) and prescribed
medications to phenotypic outcome, direction of effect, and adverse-drug-reaction
type. The active architecture is a **Two-Tower Graph Neural Network** built on
**GATv2** (PyTorch Geometric):

- **Drug tower** — molecular graphs derived from canonical SMILES via RDKit.
- **Genotype tower** — variant-topology graphs validated against GRCh38.
- The two towers are fused into multi-task prediction heads.

*Spanish version → [`README_ESP.md`](README_ESP.md).*

---

## Key features

- **Two-Tower GATv2 GNN** — attention-based encoders over drug-molecule and
  variant-topology graphs with multi-task heads for phenotype category,
  direction of effect, and ADR type.
- **Offline graph library** — drug and variant graphs are built once and
  cached to disk; training and inference lazy-load. Designed for
  compute-constrained environments.
- **FastAPI inference service** — Pydantic-typed request / response,
  auto-generated OpenAPI docs, health / readiness probes, lazy model loading.
- **Optuna integration** — hyperparameter search with per-trial pruning and a
  discriminated `OptunaSpec` union (categorical / int / float / log) parsed
  from `models.toml`.
- **Pydantic v2 at every boundary** — `Drug`, `Variant`, `StarAllele`,
  `Genotype`, `PredictionRequest`, … back every external interface;
  configuration via `pydantic-settings`.
- **NGS pipeline** — FastQC → BWA-MEM → Picard → Freebayes → VEP, argv-based
  subprocesses (no `shell=True`).

---

## Documentation

| Document | Purpose |
| --- | --- |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Module map, data flow, conventions. |
| [`docs/QUICK_START.md`](docs/QUICK_START.md) | Install, build the library, train, serve, query. |
| [`docs/LIBRARY_BUILDER.md`](docs/LIBRARY_BUILDER.md) | Input contracts, CLI flags, resume support. |
| [`docs/MEMORY_OPTIMIZATION.md`](docs/MEMORY_OPTIMIZATION.md) | OOM avoidance during training and Optuna. |
| [`docs/CODE_QUALITY.md`](docs/CODE_QUALITY.md) | Style guide (Zen of Python + SOLID). |
| [`docs/OPTIMIZATION_SUMMARY.md`](docs/OPTIMIZATION_SUMMARY.md) | Cross-cutting improvements and their rationale. |
| [`Ref.md`](Ref.md) | Phased refactor plan and history. |
| [`CLAUDE.md`](CLAUDE.md) | Working conventions for AI assistants. |

---

## Requirements

- **OS:** Linux (developed on Debian 13). WSL2 works on Windows; macOS is untested.
- **Python:** **3.14** (pinned in `.python-version`). Managed with
  [`uv`](https://github.com/astral-sh/uv); `uv.lock` is the source of truth.
- **GPU:** CUDA-capable card recommended; CPU is sufficient for inference and
  small training runs.
- **External tools** (only for the NGS pipeline): `samtools`, `bwa`, `fastp`,
  `fastqc`, `picard`, `freebayes`, `vcftools`, `vep`.

---

## Install

```bash
git clone https://github.com/Aderfi/Pharmagen.git
cd Pharmagen
uv sync --extra dev
source .venv/bin/activate
```

---

## At a glance

```bash
# Build the offline graph library (one-time, resumable)
python -m src.data.library

# Standard training
python main.py --mode train --model TwoTowerGAT --input train_data/train_data.tsv

# Hyperparameter search
python main.py --mode train --optuna --optuna-trials 50 --optuna-epochs 30

# Inference (FastAPI)
uvicorn src.api.main:app --reload          # → http://localhost:8000/docs

# Interactive CLI menu
python main.py
```

Programmatic prediction:

```python
from fastapi.testclient import TestClient
from src.api.main import create_app

client = TestClient(create_app())
client.post("/v1/predict", json={
    "drug_cid": 2244,
    "allele": {"gene": {"symbol": "CYP2D6"}, "allele": "4"},
}).json()
```

---

## Project layout

```
src/
├── api/         FastAPI inference service (routers, schemas, DI)
├── core/        Cross-cutting: exceptions, logging, validators
├── config/      Pydantic Settings + TOML data files
├── domain/      Pydantic v2 domain models
├── data/        Loading, cleaning, normalization, library builder
├── genomics/    NGS pipeline, reference genome, star alleles, variants
├── interface/   Console UI and interactive CLI
├── library/     On-disk graph cache (drugs/, gene_graphs/)
├── model/       GATv2 architecture, training, engine
└── pipeline.py  Training orchestrator
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full module map.

---

## Testing

```bash
pytest tests/unit/ -q                # 231 unit tests
pytest tests/unit/api/               # FastAPI tests
pytest tests/unit/domain/ -v         # domain models
```

Coverage is enabled by default via `pyproject.toml`
(`addopts = ["--cov=src", "--cov-report=term-missing", ...]`).

---

## License

GPLv3 — see [`LICENSE`](LICENSE).

## Author

Adrim Hamed Outmani — [@Aderfi](https://github.com/Aderfi)
