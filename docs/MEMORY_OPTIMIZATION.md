# Memory Optimization

Guidance for keeping Pharmagen training and inference inside RAM / VRAM
budgets, especially during Optuna hyperparameter searches.

## Hardware budgets

| Workload                                  | RAM       | VRAM (GPU) |
| ----------------------------------------- | --------- | ---------- |
| Inference (FastAPI)                       |  4 GB     |  2 GB      |
| Training, small datasets (< 5 000 rows)   |  8 GB     |  4 GB      |
| Training, medium datasets (5 000–20 000)  | 16 GB     |  8 GB      |
| Training, large datasets (> 20 000)       | 32 GB     | 16 GB      |
| Optuna search                             |  +50 %    |  +50 %     |

CPU-only training is supported but slow; the GATv2 towers benefit
significantly from a CUDA-capable card.

## Causes of OOM and what to do

### 1. Memory accumulates across Optuna trials

The training pipeline already enforces the safest settings while a study is
running — they're listed here so you don't fight them:

- `preload_ram=False` on the dataset.
- `num_workers=0` on the `DataLoader` (no multiprocessing pin-memory
  duplication).
- Aggressive `gc.collect()` + `torch.cuda.empty_cache()` between trials.

If a trial still OOMs:

```bash
python main.py --mode train --optuna --optuna-trials 20 --optuna-epochs 20
```

…then bisect from there. The `OptunaTrialTrainer`
(`src/model/training/optuna_trainer.py`) reports the failure to Optuna and
prunes the trial rather than crashing the whole study.

### 2. Dataset preloading on large corpora

`DoubleTowerDataset(df=large_df, preload_ram=True)` materializes every
drug + variant graph in RAM. The pipeline picks `preload_ram` automatically
based on dataset size (`PRELOAD_THRESHOLD = 10_000` rows). If you call the
dataset directly, prefer:

```python
from src.data.datasets import DoubleTowerDataset

ds = DoubleTowerDataset(df=large_df, preload_ram=False)
```

Lazy loading reads each `.pt` graph from `data/library/`
(via `Settings.paths.library`) on demand and is the default for training
runs above the threshold.

### 3. Batch size too large for VRAM

The TwoTowerGAT backbone is light, but PyG batching is variable-size — a
batch with many large variant graphs can spike VRAM unexpectedly. Start at
`batch_size=16` and increase only if memory headroom is comfortable.

### 4. Gradient accumulation as a workaround

If you need an effective batch size of 64 on a 4 GB GPU, accumulate four
batches of 16. The `StandardTrainer`
(`src/model/training/standard.py`) honours the `accumulation_steps` field on
`ModelConfig.params` if present in `models.toml`.

## Practical checks

- **Inspect the configured model.** `from src.config import get_model_config;
  get_model_config("TwoTowerGAT")` shows the resolved params used at runtime.
- **Validate the data first.** `DataValidator`
  (`src/core/validation.py`) flags missing columns, NaNs, and class imbalance
  before training starts — much cheaper than discovering them after epoch 1.
- **Catch `out of memory` early.** `src/pipeline.py` already wraps the
  training loop in a clean `except RuntimeError` that fires
  `PharmagenMemoryError` (defined in `src/core/exceptions.py`) with an
  actionable message.

## Troubleshooting checklist

`CUDA out of memory`:

1. Reduce `batch_size` by 50 %.
2. Make sure `preload_ram=False` (the pipeline does this automatically above
   `PRELOAD_THRESHOLD` rows).
3. Set `num_workers=0`.
4. Fall back to CPU (`PHARMAGEN_DEVICE=cpu`) if the GPU is too small.

RAM creeps over the course of training:

1. Check for an unbounded cache in a custom callback / encoder.
2. Run `gc.collect()` between epochs (`StandardTrainer` does this every 50
   batches by default).

## See also

- [`docs/QUICK_START.md`](QUICK_START.md) for the standard training recipe.
- [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) for the trainer split.
- `src/core/exceptions.py` for the `PharmagenMemoryError` hierarchy.
