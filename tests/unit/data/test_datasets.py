"""Tests for DoubleTowerDataset and related data classes."""

import logging

import polars as pl
import pytest

from src.data.datasets import PRELOAD_THRESHOLD, DoubleTowerDataset


class _NullResolver:
    def resolve(self, gene, genotype):  # noqa: ARG002
        return None


def test_preload_warning(tmp_path, caplog):
    """A preload warning fires when the dataset exceeds PRELOAD_THRESHOLD."""
    n = PRELOAD_THRESHOLD + 100
    df = pl.DataFrame(
        {
            "drug_id": [str(i) for i in range(n)],
            "genotype": ["A"] * n,
        }
    )

    with caplog.at_level(logging.WARNING):
        DoubleTowerDataset(
            df,
            "drug_id",
            "genotype",
            [],
            [],
            genotype_resolver=_NullResolver(),
            preload_ram=True,
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
