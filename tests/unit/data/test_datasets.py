"""Tests for DoubleTowerDataset and related data classes."""

import logging

import polars as pl

from src.data.datasets import PRELOAD_THRESHOLD, DoubleTowerDataset
from src.data.genotype import GenotypeGraphBuilder, HgvsCatalog


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
            genotype_builder=GenotypeGraphBuilder(HgvsCatalog(), {}),
            preload_ram=True,
        )

    assert "may cause OOM" in caplog.text
