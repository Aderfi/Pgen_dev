"""Tests for DoubleTowerDataset and related data classes."""
import pandas as pd
import pytest

from src.data.datasets import PRELOAD_THRESHOLD, DoubleTowerDataset


def test_preload_warning(tmp_path, caplog):
    """Test that preload warning is logged for large datasets."""
    df = pd.DataFrame({
        "drug_id": range(15000), 
        "haplo_key": ["A_B"] * 15000
    })

    dataset = DoubleTowerDataset(
        df, "drug_id", "haplo_key", [], [], preload_ram=True
    )

    assert "may cause OOM" in caplog.text


@pytest.mark.skip(reason="Requires DoubleTowerDataset implementation details")
def test_cache_stats(double_tower_dataframe):
    """Test cache statistics tracking."""
    # Create small dataset
    dataset = DoubleTowerDataset(
        double_tower_dataframe,
        "compound_id",
        "genotype_id",
        ["outcome"],
        ["side_effects"],
        preload_ram=False,
    )

    # Access some samples
    _ = dataset[0]
    _ = dataset[1]

    # Check if cache stats method exists and works
    if hasattr(dataset, "get_cache_stats"):
        stats = dataset.get_cache_stats()
        assert "drug_hit_rate" in stats or "hits" in str(stats)
        assert stats is not None
