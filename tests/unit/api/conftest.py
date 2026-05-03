"""Shared fixtures for API tests."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())
