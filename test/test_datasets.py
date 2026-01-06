# tests/test_datasets.py
import pytest
from src.data.datasets import PRELOAD_THRESHOLD, DoubleTowerDataset

def test_preload_warning(tmp_path, caplog):
    """Test that preload warning is logged for large datasets."""
    df = pd.DataFrame({"drug_id": range(15000), "haplo_key": ["A_B"] * 15000})

    dataset = DoubleTowerDataset(
        df, "drug_id", "haplo_key", [], [], preload_ram=True
    )

    assert "may cause OOM" in caplog. text


def test_cache_stats():
    """Test cache statistics tracking."""
    # Create small dataset
    dataset = DoubleTowerDataset(...)

    # Access some samples
    _ = dataset[0]
    _ = dataset[1]

    stats = dataset.get_cache_stats()
    assert "drug_hit_rate" in stats
    assert stats["drug_hits"] + stats["drug_misses"] > 0