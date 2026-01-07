# Pharmagen Utility Scripts

This directory contains utility and diagnostic scripts for Pharmagen.

## Structure

```
scripts/
├── sanity_check.py             # End-to-end sanity check for model pipeline
└── utilities/                  # Diagnostic and maintenance utilities
    ├── check_drugs_lib.py      # Verify drug graph file integrity
    ├── check_df_format.py      # Validate DataFrame format
    ├── genome_build.py         # Build genome graphs from library
    ├── test_DoubleTowerDataset.py      # DoubleTowerDataset diagnostics
    ├── test_clean_and_prepare_data.py  # Data cleaning diagnostics
    ├── test_graph_consistency.py       # Graph consistency checks
    └── debug_geno_key.py       # Genotype key debugging
```

## Scripts

### sanity_check.py
Performs an end-to-end sanity check of the model pipeline including:
- Data loading
- Dataset initialization
- Model creation
- Forward pass
- Loss calculation

**Usage:**
```bash
python scripts/sanity_check.py
```

### utilities/check_drugs_lib.py
Scans the drug library for graphs with incorrect edge dimensions.

**Usage:**
```bash
python scripts/utilities/check_drugs_lib.py
```

### utilities/check_df_format.py
Validates DataFrame format and checks for incorrect genotype formatting.

**Usage:**
```bash
python scripts/utilities/check_df_format.py
```

### utilities/genome_build.py
Builds genome graphs from the genomic library parquet file. Creates:
- JSON graph representations
- GraphML exports for visualization
- Linear genome plots
- PyTorch Geometric data objects

**Usage:**
```bash
python scripts/utilities/genome_build.py
```

## Notes

These scripts are for diagnostic and maintenance purposes, not automated tests.
For automated testing, see the `tests/` directory.
