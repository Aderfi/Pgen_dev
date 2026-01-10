# 💊 Pharmagen

Author: Adrim Hamed Outmani (@Aderfi)

> **Pharmacogenetic Prediction and Therapeutic Efficacy via Deep Learning.**

![Python Version](https://img.shields.io/badge/python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![Status](https://img.shields.io/badge/status-Beta-orange.svg)

**Pharmagen** is an advanced bioinformatics software suite designed to predict phenotypic outcomes, direction of effect, and adverse drug reaction (ADR) types based on a patient's genomic profile (Gene/Allele) and prescribed medications.

The system's core utilizes a customized **DeepFM (Deep Factorization Machines)** architecture, which combines the deep learning capabilities of Transformers with the efficiency of Factorization Machines in capturing sparse variable interactions.

---

## 🚀 Key Features

- **Hybrid Architecture:** DeepFM model featuring dynamic **embeddings** and **attention mechanisms** to capture complex drug-gene interactions.
- **Flexible Inference:**
  - **Interactive Mode (CLI):** Optimized for rapid, individual queries.
  - **Batch Mode:** For processing large patient datasets via CSV/TSV files.
- **Automated Optimization:** Native integration with **Optuna** for automated hyperparameter tuning and framework optimization.
- **Modular Configuration:** A TOML-based system (`models.toml`, `config.toml`) that allows users to define new architectures and parameters without modifying the core source code.
- **Bioinformatics Pipeline:** End-to-end processing from genomic sequencing data (VCF files) and ATC code mapping to final clinical prediction.

---

## 📋 Prerequisites

- **Operating System:** Fully compatible with **Debian 13 (Trixie)** and **Windows 11**.
- **Python:** Version **3.10** (Strictly recommended).
- **Virtual Environment:** Use of `venv` or `conda` is highly encouraged (preferably `venv`).
- **Hardware Requirements:**
    * [Add CPU/GPU/RAM requirements here]

---

## 📚 Documentation

- **[Memory Optimization Guide](docs/MEMORY_OPTIMIZATION.md)** – Guidelines for preventing OOM (Out of Memory) errors and implementing memory management best practices.
- **[Code Quality Guidelines](docs/CODE_QUALITY.md)** – Project coding standards and implementation of SOLID principles.

---

## 🛠️ Installation and Setup

Pharmagen includes an automated configuration assistant for streamlined deployment.

### 1. Clone the repository

```bash
git clone [https://github.com/Aderfi/Pharmagen.git](https://github.com/Aderfi/Pharmagen.git)
cd Pharmagen
