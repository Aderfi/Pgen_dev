# Pharmagen

> **Predicción farmacogenética y eficacia terapéutica mediante deep learning.**

![Python](https://img.shields.io/badge/python-3.14-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![PyG](https://img.shields.io/badge/PyTorch%20Geometric-2.5+-3c3c3c.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.135+-009688.svg)
![Pydantic](https://img.shields.io/badge/Pydantic-v2-e92063.svg)
![License](https://img.shields.io/badge/license-GPLv3-blue.svg)
![Status](https://img.shields.io/badge/status-Beta-orange.svg)

Pharmagen mapea el perfil genómico del paciente (gen / alelo estrella) y las
medicaciones prescritas a resultados fenotípicos, dirección del efecto y tipos
de reacción adversa al fármaco. La arquitectura activa es una **Red Neuronal
de Grafos Two-Tower** basada en **GATv2** (PyTorch Geometric):

- **Torre de fármaco** — grafos moleculares derivados de SMILES canónicos
  mediante RDKit.
- **Torre de genotipo** — grafos de topología de variantes validados frente
  a GRCh38.
- Ambas torres se fusionan en cabezas multi-tarea de predicción.

*English version → [`README.md`](README.md).*

---

## Características principales

- **GNN Two-Tower con GATv2** — codificadores basados en atención sobre
  grafos fármaco-molécula y grafos de variantes, con cabezas multi-tarea
  para categoría fenotípica, dirección del efecto y tipo de reacción
  adversa.
- **Librería de grafos offline** — los grafos de fármacos y variantes se
  construyen una sola vez y se cachean en disco; entrenamiento e inferencia
  los cargan de forma perezosa. Pensado para entornos con cómputo limitado.
- **Servicio de inferencia FastAPI** — request/response tipados con
  Pydantic, OpenAPI auto-generada, sondas health/ready, carga perezosa del
  modelo.
- **Integración con Optuna** — búsqueda de hiperparámetros con pruning por
  trial y una unión discriminada `OptunaSpec` (categorical / int / float /
  log) parseada desde `models.toml`.
- **Pydantic v2 en cada frontera** — `Drug`, `Variant`, `StarAllele`,
  `Genotype`, `PredictionRequest`, … respaldan toda interfaz externa;
  configuración mediante `pydantic-settings`.
- **Pipeline NGS** — FastQC → BWA-MEM → Picard → Freebayes → VEP, con
  subprocesos basados en `argv` (sin `shell=True`).

---

## Documentación

> Toda la documentación técnica detallada está en inglés (decisión
> tomada en la Fase 6 del refactor para evitar mantener dos versiones).

| Documento | Propósito |
| --- | --- |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Mapa de módulos, flujo de datos, convenciones. |
| [`docs/QUICK_START.md`](docs/QUICK_START.md) | Instalación, construcción de la librería, entrenamiento, servicio. |
| [`docs/LIBRARY_BUILDER.md`](docs/LIBRARY_BUILDER.md) | Contratos de entrada, flags de la CLI, soporte de resume. |
| [`docs/MEMORY_OPTIMIZATION.md`](docs/MEMORY_OPTIMIZATION.md) | Evitar OOM durante entrenamiento y Optuna. |
| [`docs/CODE_QUALITY.md`](docs/CODE_QUALITY.md) | Guía de estilo (Zen of Python + SOLID). |
| [`docs/OPTIMIZATION_SUMMARY.md`](docs/OPTIMIZATION_SUMMARY.md) | Mejoras transversales y su justificación. |
| [`Ref.md`](Ref.md) | Plan e historia del refactor por fases. |
| [`CLAUDE.md`](CLAUDE.md) | Convenciones de trabajo para asistentes de IA. |

---

## Requisitos

- **SO:** Linux (desarrollado en Debian 13). WSL2 funciona en Windows;
  macOS no probado.
- **Python:** **3.14** (fijado en `.python-version`). Gestionado con
  [`uv`](https://github.com/astral-sh/uv); `uv.lock` es la fuente de
  verdad.
- **GPU:** Tarjeta con CUDA recomendada; CPU es suficiente para
  inferencia y entrenamientos pequeños.
- **Herramientas externas** (sólo para el pipeline NGS): `samtools`,
  `bwa`, `fastp`, `fastqc`, `picard`, `freebayes`, `vcftools`, `vep`.

---

## Instalación

```bash
git clone https://github.com/Aderfi/Pharmagen.git
cd Pharmagen
uv sync --extra dev
source .venv/bin/activate
```

---

## Vista rápida

```bash
# Construir la librería offline (una sola vez, reanudable)
python -m src.data.library

# Entrenamiento estándar
python main.py --mode train --model TwoTowerGAT --input train_data/train_data.tsv

# Búsqueda de hiperparámetros
python main.py --mode train --optuna --optuna-trials 50 --optuna-epochs 30

# Inferencia (FastAPI)
uvicorn src.api.main:app --reload          # → http://localhost:8000/docs

# Menú CLI interactivo
python main.py
```

Predicción programática:

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

## Estructura del proyecto

```
src/
├── api/         Servicio de inferencia FastAPI (routers, schemas, DI)
├── core/        Transversal: excepciones, logging, validadores
├── config/      Pydantic Settings + archivos TOML de datos
├── domain/      Modelos de dominio Pydantic v2
├── data/        Carga, limpieza, normalización, library builder
├── genomics/    Pipeline NGS, genoma de referencia, alelos estrella, variantes
├── interface/   UI de consola y CLI interactiva
├── model/       Arquitectura GATv2, entrenamiento, engine (engine/base.py es el bootstrap compartido)
└── pipeline.py  Orquestador de entrenamiento

data/
├── library/     Caché de grafos en disco (drugs/, gene_graphs/) — Settings.paths.library
├── dicts/       star_alleles.tsv y otras tablas estáticas
└── …            raw/, processed/, ref_genome/

scripts/         Utilidades de visualización / inspección
.github/         Workflow de CI (ruff + pytest en push / PR)
```

Ver [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) para el mapa completo de módulos.

---

## Tests

```bash
uv run pytest tests/unit -q          # suite unitaria completa (~240 tests)
uv run pytest tests/integration -q   # smoke tests del pipeline
uv run pytest tests/unit/api -q      # tests de FastAPI
uv run pytest tests/unit/domain -v   # modelos de dominio
```

La CI replica esto en cada push y PR vía `.github/workflows/ci.yml`.

La cobertura está activada por defecto vía `pyproject.toml`
(`addopts = ["--cov=src", "--cov-report=term-missing", ...]`).

---

## Licencia

GPLv3 — ver [`LICENSE`](LICENSE).

## Autor

Adrim Hamed Outmani — [@Aderfi](https://github.com/Aderfi)
