# Quick Start Guide - Optimized Pharmagen

## Installation

```bash
git clone https://github.com/Aderfi/Pharmagen_development
cd Pharmagen_development
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

## Basic Training

### Standard Training (Recommended)

```python
from src.pipeline import train_pipeline

# The pipeline now automatically:
# - Validates configuration
# - Checks data quality
# - Monitors memory usage
# - Adjusts workers based on dataset size
# - Handles errors gracefully

train_pipeline(
    model_name="TwoTowerGAT",
    csv_path="train_data/train_data.tsv",
    epochs=50,
    batch_size=32
)
```

### Training with Memory Monitoring

```python
from src.pipeline import train_pipeline
from src.utils.memory import MemoryMonitor

# Monitor memory before and after
MemoryMonitor.log_memory_stats("Before training")
train_pipeline("TwoTowerGAT", "data.tsv", epochs=30, batch_size=16)
MemoryMonitor.log_memory_stats("After training")
```

## Optuna Hyperparameter Optimization

### Basic Optuna Study

```python
from src.modeling.engine.tuner import run_optuna_study

# Now includes automatic:
# - Memory cleanup after each trial
# - OOM prevention
# - Trial-level memory logging
# - Proper resource cleanup

run_optuna_study(
    model_name="TwoTowerGAT",
    csv_path="train_data/train_data.tsv",
    n_trials=50  # Start with 20-50, increase if memory allows
)
```

### Advanced Optuna with Custom Settings

```python
from src.modeling.engine.tuner import PGenTuner

tuner = PGenTuner(
    model_name="TwoTowerGAT",
    csv_path="train_data/train_data.tsv",
    random_seed=711,
    max_batch_size=64  # Prevent trials from using excessive memory
)

# Single job to avoid memory conflicts
study = tuner.run_tuning(n_trials=50, n_jobs=1)

print(f"Best params: {study.best_params}")
print(f"Best value: {study.best_value}")
```

## Memory-Constrained Scenarios

### Small GPU (4-6GB VRAM)

```python
# Use smaller batch size and disable preloading
train_pipeline(
    model_name="TwoTowerGAT",
    csv_path="data.tsv",
    batch_size=8,  # Small batch
    epochs=50
)
```

### Large Dataset (>10k samples)

```python
# Pipeline automatically sets preload_ram=False for large datasets
# and adjusts worker count

train_pipeline(
    model_name="TwoTowerGAT",
    csv_path="large_data.tsv",
    batch_size=32,
    epochs=50
)
```

### CPU-Only Training

```python
import torch

# Force CPU (automatic if CUDA unavailable)
# Pipeline will adjust settings accordingly
train_pipeline(
    model_name="TwoTowerGAT",
    csv_path="data.tsv",
    batch_size=16,
    epochs=30
)
```

## Configuration

### Model Configuration (models.toml)

```toml
[TwoTowerGAT]
features = ["compound_id", "haplo_key"]
targets = ["Phenotype_Effect_Outcome"]

[TwoTowerGAT.params]
learning_rate = 0.001
batch_size = 32
hidden_dim = 128
embedding_dim = 64
num_layers = 3
heads = 4
dropout = 0.1

[TwoTowerGAT.optuna]
# Will be validated automatically
learning_rate = ["log", 1e-4, 1e-2]
batch_size = ["categorical", 16, 32, 64]
hidden_dim = ["categorical", 64, 128, 256]
dropout = ["float", 0.1, 0.3]
```

### Validation Happens Automatically

```python
from src.config.manager import get_model_config

# This now includes automatic validation
try:
    config = get_model_config("TwoTowerGAT")
    # Configuration validated successfully
except ConfigurationError as e:
    print(f"Config error: {e}")
    # Error message will be specific and actionable
```

## Error Handling

### Graceful Error Messages

```python
from src.pipeline import train_pipeline
from src.utils.exceptions import ConfigurationError, DataError, MemoryError

try:
    train_pipeline("MyModel", "data.tsv")
except ConfigurationError as e:
    print(f"Configuration issue: {e}")
    # e.g., "Model 'MyModel' missing required param 'learning_rate'"
    
except DataError as e:
    print(f"Data issue: {e}")
    # e.g., "Dataset missing required column 'Phenotype_Effect_Outcome'"
    
except MemoryError as e:
    print(f"Memory issue: {e}")
    # e.g., "Insufficient GPU memory: 2048MB available, 4096MB required"
```

## Memory Monitoring

### Check Available Memory

```python
from src.utils.memory import MemoryMonitor
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Check if enough memory for operation
has_memory = MemoryMonitor.check_memory_available(
    required_mb=4096,  # Need 4GB
    device=device,
    raise_error=True  # Raise if insufficient
)
```

### Estimate Requirements

```python
from src.utils.memory import estimate_model_memory_mb, estimate_batch_memory_mb

# Estimate model memory
model_mem = estimate_model_memory_mb(
    num_parameters=5_000_000,  # 5M parameters
    optimizer_factor=2.0  # Adam needs 2x for state
)
print(f"Model will need ~{model_mem:.0f}MB")

# Estimate batch memory
batch_mem = estimate_batch_memory_mb(
    batch_size=32,
    avg_nodes_per_graph=50,
    node_features=25
)
print(f"Batch will need ~{batch_mem:.0f}MB")
```

### Manual Cleanup

```python
from src.utils.memory import MemoryMonitor
import torch

device = torch.device("cuda:0")

# After intensive operation
MemoryMonitor.clear_memory(device=device, aggressive=True)

# Check memory was freed
MemoryMonitor.log_memory_stats("After cleanup")
```

## Validation

### Validate Configuration Before Training

```python
from src.utils.validation import ConfigValidator, DataValidator
import pandas as pd

# Validate model config
config = {"features": ["drug"], "targets": ["outcome"], "params": {"learning_rate": 0.001}}
ConfigValidator.validate_model_config(config, "test_model")

# Validate data
df = pd.read_csv("data.tsv", sep="\t")
ConfigValidator.validate_data_columns(
    df.columns.tolist(),
    required_features=["compound_id"],
    required_targets=["Phenotype_Effect_Outcome"]
)

# Check data quality
DataValidator.check_missing_values(df, ["compound_id", "haplo_key"])
DataValidator.check_class_balance(df, "Phenotype_Effect_Outcome")
```

## Common Workflows

### 1. Initial Training

```python
from src.pipeline import train_pipeline

# Start small to test
train_pipeline(
    model_name="TwoTowerGAT",
    csv_path="data.tsv",
    batch_size=16,
    epochs=10  # Few epochs to test
)
```

### 2. Hyperparameter Optimization

```python
from src.modeling.engine.tuner import run_optuna_study

# Run optimization
run_optuna_study(
    model_name="TwoTowerGAT",
    csv_path="data.tsv",
    n_trials=20  # Start small
)

# Check results in: reports/optuna_reports/
```

### 3. Production Training with Best Params

```python
# After Optuna, update models.toml with best params
# Then run full training

train_pipeline(
    model_name="TwoTowerGAT",
    csv_path="data.tsv",
    batch_size=32,  # From Optuna
    epochs=100  # Full training
)
```

### 4. Prediction

```python
from src.modeling.engine.predictor import PGenPredictor
import pandas as pd

predictor = PGenPredictor("TwoTowerGAT")
results = predictor.predict_file("new_patients.csv")

# Save predictions
pd.DataFrame(results).to_csv("predictions.csv", index=False)
```

## Troubleshooting

### Out of Memory?

See [Memory Optimization Guide](MEMORY_OPTIMIZATION.md) for detailed solutions.

Quick fixes:
1. Reduce batch size: `batch_size=16` or `8`
2. Ensure `preload_ram=False` for large datasets
3. Use `num_workers=0` during Optuna
4. Check memory: `MemoryMonitor.log_memory_stats()`

### Configuration Errors?

```python
from src.config.manager import get_model_config

try:
    config = get_model_config("MyModel")
except Exception as e:
    print(f"Config error: {e}")
    # Fix the issue in models.toml
```

### Data Errors?

```python
from src.utils.validation import DataValidator
import pandas as pd

df = pd.read_csv("data.csv")
missing = DataValidator.check_missing_values(df, df.columns.tolist())
print(f"Missing values: {missing}")
```

## Best Practices

1. **Start Small**: Begin with small batch sizes and few epochs
2. **Monitor Memory**: Use `MemoryMonitor.log_memory_stats()` at key points
3. **Validate Early**: Configuration and data validation happen automatically
4. **Check Logs**: Review `logs/` directory for detailed information
5. **Use Documentation**: Refer to guides in `docs/` directory

## Additional Resources

- [Memory Optimization Guide](MEMORY_OPTIMIZATION.md)
- [Code Quality Guidelines](CODE_QUALITY.md)
- [Optimization Summary](OPTIMIZATION_SUMMARY.md)

## Getting Help

1. Check the documentation in `docs/`
2. Review error messages (now more detailed and actionable)
3. Enable debug logging: `logging.basicConfig(level=logging.DEBUG)`
4. Open an issue on GitHub with:
   - System specs
   - Configuration used
   - Full error traceback
