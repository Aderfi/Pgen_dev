"""
Shared pytest fixtures for Pharmagen test suite.
"""
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import torch


@pytest.fixture
def sample_dataframe():
    """Basic DataFrame for general testing."""
    return pd.DataFrame({
        "gene_id": ["G1", "G2", "G3", "G1"],
        "drug_id": ["D1", "D2", "D1", "D3"],
        "outcome": [0, 1, 0, 1],
    })


@pytest.fixture
def double_tower_dataframe():
    """DataFrame structure for DoubleTowerDataset testing."""
    return pd.DataFrame({
        "compound_id": ["DRUG_001", "DRUG_002", "DRUG_003"] * 10,
        "genotype_id": ["CYP2D6_*1_*2", "CYP2C19_*1_*17", "CYP3A4_*1_*1"] * 10,
        "outcome": [0, 1, 0] * 10,
        "side_effects": ["SE1|SE2", "SE2", "SE1|SE3"] * 10,
    })


@pytest.fixture
def model_params():
    """Default parameters for DeepFM model initialization."""
    return {
        "n_features": {"gene_id": 100, "drug_id": 50},
        "target_dims": {"outcome": 1, "side_effects": 10},
        "embedding_dim": 16,
        "hidden_dim": 32,
        "dropout_rate": 0.1,
        "n_layers": 2,
        "attention_dim_feedforward": 64,
        "attention_dropout": 0.1,
        "num_attention_layers": 1,
        "use_batch_norm": True,
        "use_layer_norm": False,
        "activation_function": "relu",
        "fm_dropout": 0.1,
        "fm_hidden_layers": 1,
        "fm_hidden_dim": 16,
        "embedding_dropout": 0.1,
    }


@pytest.fixture
def dummy_graph():
    """Create a dummy PyTorch Geometric graph for testing."""
    try:
        from torch_geometric.data import Data

        # Simple graph with 5 nodes and some edges
        x = torch.randn(5, 10)  # 5 nodes, 10 features each
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        edge_attr = torch.randn(4, 7)  # 4 edges, 7 features each

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    except ImportError:
        pytest.skip("PyTorch Geometric not installed")


@pytest.fixture
def mock_config():
    """Mock configuration for pipeline testing."""
    return {
        "features": ["gene_id", "drug_id"],
        "targets": ["outcome"],
        "drug_col": "compound_id",
        "haplo_col": "genotype_id",
        "params": {
            "batch_size": 16,
            "learning_rate": 0.001,
            "embedding_dim": 8,
            "hidden_dim": 16,
            "dropout_rate": 0.1,
            "n_layers": 1,
            "weight_decay": 0.0,
            "early_stopping_patience": 3,
        },
    }


@pytest.fixture
def temp_library(tmp_path):
    """Create a temporary directory for library testing."""
    library_dir = tmp_path / "library"
    library_dir.mkdir()
    return library_dir


@pytest.fixture
def device():
    """Get appropriate torch device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def mock_encoder():
    """Create a mock sklearn LabelEncoder."""
    encoder = MagicMock()
    encoder.classes_ = np.array(["A", "B", "C"])
    encoder.transform.return_value = np.array([0, 1, 2])
    encoder.inverse_transform.return_value = np.array(["A", "B", "C"])
    return encoder
