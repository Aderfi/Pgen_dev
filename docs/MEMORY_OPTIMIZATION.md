# Pharmagen Memory Optimization Guide

## Overview

This guide provides recommendations for preventing Out-Of-Memory (OOM) errors when training models, especially during Optuna hyperparameter optimization.

## Memory Requirements

### Minimum Requirements

- **Training (small datasets <5k samples)**: 8GB RAM, 4GB VRAM (GPU)
- **Training (medium datasets 5-20k samples)**: 16GB RAM, 8GB VRAM
- **Training (large datasets >20k samples)**: 32GB+ RAM, 16GB+ VRAM
- **Optuna optimization**: +50% memory over normal training

### Estimating Memory Usage

```python
from src.utils.memory import estimate_model_memory_mb, estimate_batch_memory_mb

# Estimate model memory
num_params = sum(p.numel() for p in model.parameters())
model_mem = estimate_model_memory_mb(num_params)
print(f"Model memory: {model_mem:.1f}MB")

# Estimate batch memory
batch_mem = estimate_batch_memory_mb(
    batch_size=32,
    avg_nodes_per_graph=50,
    node_features=25,
    num_graphs=2
)
print(f"Batch memory: {batch_mem:.1f}MB")
```

## Common OOM Scenarios and Solutions

### 1. OOM During Optuna Optimization

**Problem**: Memory accumulates across trials.

**Solutions**:

```python
# ✅ GOOD: Proper memory management
tuner = PGenTuner(
    model_name="TwoTowerGAT",
    csv_path="data.tsv",
    max_batch_size=64  # Limit batch size
)
study = tuner.run_tuning(n_trials=50, n_jobs=1)
```

**Key Settings**:
- `preload_ram=False` (always during Optuna)
- `num_workers=0` during optimization
- Automatic cleanup after each trial

### 2. Dataset Preloading

```python
# ❌ BAD: Preloading large dataset
dataset = DoubleTowerDataset(df=large_df, preload_ram=True)

# ✅ GOOD: Lazy loading
dataset = DoubleTowerDataset(df=large_df, preload_ram=False)
```

## Best Practices

1. **Monitor Memory**: Use `MemoryMonitor.log_memory_stats()`
2. **Start Small**: Begin with `batch_size=16`, increase if memory allows
3. **Clear Caches**: Periodic cleanup every 50-100 batches
4. **Validate Early**: Check config and data before training
5. **Use Mixed Precision**: Automatic via `torch.amp.autocast`

## Troubleshooting

### "CUDA out of memory"
1. Reduce batch size by 50%
2. Set `preload_ram=False`
3. Use `num_workers=0`
4. Try CPU if GPU insufficient

### Memory grows during training
1. Enable periodic cleanup (automatic)
2. Check for unbounded caches
3. Run garbage collection

## Hardware Recommendations

- **Minimum**: 16GB RAM, 6GB VRAM
- **Recommended**: 32GB RAM, 12GB VRAM
- **Optimal**: 64GB RAM, 24GB VRAM

See full guide in repository for detailed information.
