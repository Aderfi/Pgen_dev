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

## Project status & roadmap

Pharmagen is evolving from a research prototype into maintainable, deployable
software. Two tracks run in parallel:

**v1 — research prototype (complete).**
A hybrid neural network (Factorization Machines + Transformers with attention)
trained on ClinPGx and dbSNP. It validated the core hypothesis and defined the
problem — while also exposing evaluation pitfalls (data-handling assumptions
that likely inflated early metrics), which directly motivate the rigorous,
leakage-aware evaluation planned for v2.

**v2 — current (Beta).** Two concurrent efforts:

- **Scientific — graph architecture.** Migration to a Two-Tower **GATv2** GNN
  (PyTorch Geometric): drug-molecule graphs (RDKit) and variant-topology graphs,
  capturing structure the v1 feature engineering could not.
- **Engineering — production hardening.** A full refactor to industry standards
  the first version lacked: typed data models (Pydantic v2) at every boundary,
  a FastAPI inference service, reproducible workflows, CI (ruff + pytest),
  `uv`-managed dependencies, and a rigorous, leakage-aware evaluation protocol.
  Turning a working experiment into software others can build on.

> The v1 prototype proved the science; v2 is about making it robust,
> reproducible, and deployable.

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
├── model/       GATv2 architecture, training, engine (engine/base.py is shared bootstrap)
└── pipeline.py  Training orchestrator

data/
├── library/     On-disk graph cache (drugs/, gene_graphs/) — Settings.paths.library
├── dicts/       star_alleles.tsv and other static lookups
└── …            raw/, processed/, ref_genome/

scripts/         Standalone visualisation / inspection utilities
.github/         CI workflow (ruff + pytest on push / PR)
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full module map.

---

## Testing

```bash
uv run pytest tests/unit -q          # full unit suite (~240 tests)
uv run pytest tests/integration -q   # pipeline smoke tests
uv run pytest tests/unit/api -q      # FastAPI tests
uv run pytest tests/unit/domain -v   # domain models
```

CI mirrors this on every push and PR via `.github/workflows/ci.yml`.

Coverage is enabled by default via `pyproject.toml`
(`addopts = ["--cov=src", "--cov-report=term-missing", ...]`).

---

## License

GPLv3 — see [`LICENSE`](LICENSE).

## Author

Adrim Hamed Outmani — [@Aderfi](https://github.com/Aderfi)
