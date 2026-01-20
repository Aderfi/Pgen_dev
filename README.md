**README languages &#8594;** **[ENG](#-pharmagen-eng)  /  [ESP](#-pharmagen-esp)**

---
# 💊 Pharmagen {#eng}

Author: Adrim Hamed Outmani (@Aderfi)

![Python Version](https://img.shields.io/badge/python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![Status](https://img.shields.io/badge/status-Beta-orange.svg)

> **Pharmacogenetic Prediction and Therapeutic Efficacy via Deep Learning.**

**Pharmagen** is an advanced bioinformatics software suite designed to predict phenotypic outcomes, direction of effect, and adverse drug reaction (ADR) types based on a patient's genomic profile (Gene/Allele) and prescribed medications.

The system's core utilizes a customized **DeepFM (Deep Factorization Machines)** architecture, which combines the deep learning capabilities of Transformers with the efficiency of Factorization Machines in capturing sparse variable interactions.

---

## 🚀 Key Features

- **Hybrid Architecture:** ~~DeepFM model featuring dynamic **embeddings** and **attention mechanisms** to capture complex drug-gene interactions.~~
  Pharmagen was categorical model. **Now working on a GNN structure based on GATv2**

- **Flexible Inference:**
  - **Interactive Mode (CLI):** Optimized for rapid, individual queries.
  - **Batch Mode:** For processing large patient datasets via CSV/TSV files.
- **Automated Optimization:** Native integration with **Optuna** for automated hyperparameter tuning and framework optimization.
- **Modular Configuration:** A TOML-based system (`models.toml`, `config.toml`) that allows users to define new architectures and parameters without modifying the core source code.
- **Bioinformatics Pipeline:** End-to-end processing from genomic sequencing data (VCF files) and ATC code mapping to final clinical prediction.

---

## 📋 Prerequisites

- **Operating System:** Developed on **Debian 13 (Trixie)** and **Windows 11**. 
- **Python:** Version **3.10** (Strictly recommended).
- **Virtual Environment:** Use of `venv` or `conda` is highly encouraged ****(preferably `venv` through `UV`)****. if `conda` --> `mamba`
- **Hardware Requirements:**
    * [CPU/GPU/RAM requirements]

---

## 📚 Documentation

- **Graph Library Build** &#8594; **[ENG](docs/BUILD_LIBRARY.md) / [ESP](docs/BUILD_LIBRARY_ES.md)** - Guide for 
- **[Memory Optimization Guide](docs/MEMORY_OPTIMIZATION.md)** – Guidelines for preventing OOM (Out of Memory) errors and implementing memory management best practices. --> Doc redacted by AI (Gemini) so i can have an accesible sumup
- **[Code Quality Guidelines](docs/CODE_QUALITY.md)** – Project coding standards and implementation of SOLID principles. --> Doc redacted by AI (Gemini) so i can have an accesible sumup

---

## 🛠️ Installation and Setup

Pharmagen includes an automated configuration assistant for streamlined deployment.

### 1. Clone the repository

```bash
git clone [https://github.com/Aderfi/Pharmagen.git](https://github.com/Aderfi/Pharmagen.git)
cd Pharmagen
```
---
---
---

# 💊 Pharmagen {#ESP}

Author: Adrim Hamed Outmani (@Aderfi)

![Python Version](https://img.shields.io/badge/python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![Status](https://img.shields.io/badge/status-Beta-orange.svg)


> **Predicción Farmacogenética y Eficacia Terapéutica mediante Deep Learning.**

**Pharmagen** es un software avanzado de bioinformática diseñado para predecir resultados fenotípicos, dirección del efecto y tipos de efectos adversos en pacientes basándose en su perfil genómico (Gen/Alelo) y fármacos prescritos.

El núcleo del sistema utiliza una arquitectura **DeepFM (Deep Factorization Machines)** personalizada, que combina la capacidad de aprendizaje profundo de los Transformers con la eficiencia en interacciones de variables dispersas de las Máquinas de Factorización.


## 🚀 Características Principales

- **Arquitectura Híbrida:** Modelo DeepFM con _Embeddings_ dinámicos y _Attention Mechanism_ para capturar interacciones complejas fármaco-gen.
- **Inferencia Flexible:**
  - Modo Interactivo (CLI) para consultas rápidas.
  - Modo _Batch_ para procesar grandes volúmenes de pacientes (CSV/TSV).
- **Optimización Automatizada:** Integración nativa con **Optuna** para la búsqueda de hiperparámetros.
- **Configuración Modular:** Sistema basado en archivos TOML (`models.toml`, `config.toml`) que permite definir nuevas arquitecturas sin tocar el código.
- **Pipeline Bioinformático:** Procesamiento de secuenciaciones genómicas a archivos VCF y mapeo a códigos ATC para la predicción.

---

## 📋 Requisitos Previos

- **Sistema Operativo:** Linux, macOS o Windows.
- **Python:** Versión **3.10** (Estrictamente recomendada).
- **Entorno Virtual:** Se recomienda encarecidamente usar `venv` o `conda`. (Preferiblemente con `venv`)
- **Hardware Mínimo:**
    ... 

---

## 📚 Documentación

- **[Memory Optimization Guide](docs/MEMORY_OPTIMIZATION.md)** - Prevención de errores OOM y mejores prácticas de memoria
- **[Code Quality Guidelines](docs/CODE_QUALITY.md)** - Estándares de código y principios SOLID

---

## 🛠️ Instalación y Configuración

Pharmagen incluye un asistente de configuración automatizado.

### 1. Clonar el repositorio

```bash
git clone [Pharmagen](https://github.com/Aderfi/Pharmagen)
cd pharmagen
```

