# test_graph_consistency.py

import torch

from torch. utils.data import DataLoader

from src.data.datasets import DoubleTowerDataset
from src.data.collator import DoubleTowerCollater

# Load dataset
dataset = DoubleTowerDataset(
    df=your_train_df,
    drug_col="drug_id",
    haplo_col="haplo_key",
    target_cols=["outcome"],
    multilabel_cols=[],
    preload_ram=False,
)

# Test individual samples
for i in range(10):
    sample = dataset[i]
    
    drug = sample["drug_data"]
    haplo = sample["haplo_data"]
    
    print(f"\nSample {i}:")
    print(f"  Drug: x={drug.x.shape}, edge_attr={drug.edge_attr.shape if drug.edge_attr is not None else None}")
    print(f"  Haplo: x={haplo.x.shape}, edge_attr={haplo.edge_attr. shape if haplo.edge_attr is not None else None}")
    
    # Validate
    if drug.edge_attr is None:
        print(f"  ❌ Drug graph has NO edge_attr!")
    if haplo.edge_attr is None:
        print(f"  ❌ Haplo graph has NO edge_attr!")

# Test batching
loader = DataLoader(dataset, batch_size=4, collate_fn=DoubleTowerCollater())

try:
    batch = next(iter(loader))
    print("\n✅ Batching successful!")
    print(f"  Drug batch:  {batch['drug_batch']}")
    print(f"  Haplo batch: {batch['haplo_batch']}")
except Exception as e:
    print(f"\n❌ Batching failed: {e}")