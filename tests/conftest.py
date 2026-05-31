"""Shared pytest fixtures for Pharmagen test suite."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch


@pytest.fixture
def device():
    """Appropriate torch device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def mock_encoder():
    """Mock sklearn LabelEncoder."""
    encoder = MagicMock()
    encoder.classes_ = np.array(["A", "B", "C"])
    encoder.transform.return_value = np.array([0, 1, 2])
    encoder.inverse_transform.return_value = np.array(["A", "B", "C"])
    return encoder
