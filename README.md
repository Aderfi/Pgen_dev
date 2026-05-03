**README languages →** **[ENG](#-pharmagen-eng)  /  [ESP](#-pharmagen-esp)**

---

# 💊 Pharmagen {#eng}

Author: Adrim Hamed Outmani (@Aderfi)

![Python Version](https://img.shields.io/badge/python-3.14-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.135+-009688.svg)
![Pydantic](https://img.shields.io/badge/Pydantic-v2-e92063.svg)
![Status](https://img.shields.io/badge/status-Beta-orange.svg)

> **Pharmacogenetic Prediction and Therapeutic Efficacy via Deep Learning.**

**Pharmagen** is a bioinformatics suite that maps a patient's genomic profile (gene/allele) and prescribed medications to phenotypic outcomes, direction-of-effect, and adverse-drug-reaction types.

The active architecture is a **Two-Tower Graph Neural Network** built on **GATv2** (PyTorch Geometric) — a drug tower over molecular SMILES graphs, a genotype tower over variant-topology graphs, fused into multi-task heads. (The earlier DeepFM design is being phased out.)

---

## 🚀 Key features

- **Two-Tower GATv2 GNN** — drug-molecule + genotype graphs encoded with attention; multi-task prediction heads for phenotype category, direction of effect, and ADR.
- **Offline graph library** — all drug + variant graphs are built once and stored to disk (`src/library/`); training and inference lazy-load. Critical for limited-compute environments.
- **FastAPI inference service** — typed Pydantic request/response, OpenAPI docs at `/docs`, health/ready probes, lazy model loading.
- **Optuna integration** — hyperparameter search with per-trial pruning and a discriminated `OptunaSpec` union (categorical / int / float / log) parsed from `models.toml`.
- **Pydantic v2 throughout** — domain models (`Drug`, `Variant`, `StarAllele`, `Genotype`, `PredictionRequest`, …) at every boundary; configuration via `pydantic-settings`.
- **NGS pipeline** — FastQC → BWA-MEM → Picard → Freebayes → VEP, no `shell=True`, sane subprocess plumbing.

---

## 📚 Documentation

- **[Architecture](docs/ARCHITECTURE.md)** — module map, data flow, conventions.
- **[Quick Start](docs/QUICK_START.md)** — install, build the library, train, serve, query the API.
- **[Library Builder](docs/LIBRARY_BUILDER.md)** — input contracts, CLI flags, resume support, schema.
- **[Memory Optimization](docs/MEMORY_OPTIMIZATION.md)** — OOM avoidance during training & Optuna.
- **[Code Quality](docs/CODE_QUALITY.md)** — Zen of Python + SOLID examples.
- **[Refactor plan & history](Ref.md)** — phased rewrite of `src/`.

For AI assistants: `CLAUDE.md` documents working conventions and current refactor state.

---

## 📋 Prerequisites

- **OS:** Linux (developed on Debian 13). Windows works with WSL2; macOS is untested.
- **Python:** **3.14** (pinned in `.python-version`). The project is managed with [`uv`](https://github.com/astral-sh/uv); `uv.lock` is the source of truth.
- **GPU:** CUDA-capable card recommended; CPU works for inference and small training runs.
- **External tools** (only needed for the NGS pipeline): `samtools`, `bwa`, `fastp`, `fastqc`, `picard`, `freebayes`, `vcftools`, `vep`.

---

## 🛠️ Quick install

```bash
git clone https://github.com/Aderfi/Pharmagen.git
cd Pharmagen
uv sync --extra dev
source .venv/bin/activate
```

---

## ⚡ At a glance

```bash
# Build the offline graph library (one-time, resumable)
python -m src.data.library

# Train
python main.py --mode train --model TwoTowerGAT --input train_data/train_data.tsv

# Hyperparameter search
python main.py --mode train --optuna --optuna-trials 50

# Inference API
uvicorn src.api.main:app --reload         # → http://localhost:8000/docs

# Interactive CLI menu
python main.py
```

```python
# Programmatic prediction
from fastapi.testclient import TestClient
from src.api.main import create_app

client = TestClient(create_app())
client.post("/v1/predict", json={
    "drug_cid": 2244,
    "allele": {"gene": {"symbol": "CYP2D6"}, "allele": "4"}
}).json()
```

---

## 🧪 Testing

```bash
pytest tests/unit/ -q --override-ini="addopts="     # 237 tests
```

---

---

# 💊 Pharmagen {#ESP}

Autor: Adrim Hamed Outmani (@Aderfi)

![Python Version](https://img.shields.io/badge/python-3.14-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.135+-009688.svg)
![Status](https://img.shields.io/badge/status-Beta-orange.svg)

> **Predicción farmacogenética y eficacia terapéutica mediante Deep Learning.**

**Pharmagen** es una suite bioinformática que mapea el perfil genómico del paciente (gen/alelo) y las medicaciones prescritas a resultados fenotípicos, dirección del efecto y tipos de reacciones adversas.

La arquitectura activa es una **Red Neuronal de Grafos Two-Tower** basada en **GATv2** (PyTorch Geometric) — torre de fármaco sobre grafos moleculares SMILES, torre de genotipo sobre grafos de topología de variantes, fusionadas en cabezas multi-tarea. (El diseño DeepFM previo está siendo retirado.)

---

## 🚀 Características principales

- **GNN Two-Tower con GATv2** — grafos fármaco-molécula + genotipo codificados con atención; cabezas de predicción multi-tarea.
- **Librería de grafos offline** — todos los grafos de fármacos y variantes se construyen una vez y se almacenan en disco (`src/library/`); entrenamiento e inferencia hacen lazy-load. Esencial para entornos con cómputo limitado.
- **Servicio de inferencia FastAPI** — request/response tipados con Pydantic, documentación OpenAPI en `/docs`, sondas health/ready, carga perezosa del modelo.
- **Integración Optuna** — búsqueda de hiperparámetros con pruning por trial y unión discriminada `OptunaSpec` (categorical / int / float / log) parseada desde `models.toml`.
- **Pydantic v2 en toda la base** — modelos de dominio (`Drug`, `Variant`, `StarAllele`, `Genotype`, `PredictionRequest`, …) en cada frontera; configuración vía `pydantic-settings`.
- **Pipeline NGS** — FastQC → BWA-MEM → Picard → Freebayes → VEP, sin `shell=True`, subprocesos sanos.

---

## 📚 Documentación

- **[Arquitectura (EN)](docs/ARCHITECTURE.md)** — mapa de módulos, flujo de datos, convenciones.
- **[Quick Start (EN)](docs/QUICK_START.md)** — instalar, construir la librería, entrenar, servir, consumir la API.
- **[Library Builder (EN)](docs/LIBRARY_BUILDER.md)** — contratos de entrada, flags de la CLI, soporte de resume, esquemas.
- **[Memory Optimization (EN)](docs/MEMORY_OPTIMIZATION.md)** — evitar OOM en entrenamiento y Optuna.
- **[Code Quality (EN)](docs/CODE_QUALITY.md)** — ejemplos de Zen of Python + SOLID.
- **[Plan e historia del refactor (EN)](Ref.md)** — reescritura por fases de `src/`.

> La documentación técnica está en inglés tras el refactor de localización (Fase 6). El README mantiene su estructura bilingüe.

---

## 📋 Requisitos

- **SO:** Linux (desarrollado en Debian 13). Windows funciona con WSL2; macOS no probado.
- **Python:** **3.14** (fijado en `.python-version`). Gestionado con [`uv`](https://github.com/astral-sh/uv); `uv.lock` es la fuente de verdad.
- **GPU:** Tarjeta con CUDA recomendada; CPU funciona para inferencia y entrenamientos pequeños.

---

## 🛠️ Instalación rápida

```bash
git clone https://github.com/Aderfi/Pharmagen.git
cd Pharmagen
uv sync --extra dev
source .venv/bin/activate
```

---

## ⚡ Vista rápida

```bash
# Construir la librería offline (una sola vez, resumible)
python -m src.data.library

# Entrenar
python main.py --mode train --model TwoTowerGAT --input train_data/train_data.tsv

# Búsqueda de hiperparámetros
python main.py --mode train --optuna --optuna-trials 50

# API de inferencia
uvicorn src.api.main:app --reload         # → http://localhost:8000/docs

# Menú CLI interactivo
python main.py
```
