"""Tests for /v1/models."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


class TestListModels:
    def test_returns_at_least_one(self, client: TestClient) -> None:
        r = client.get("/v1/models")
        assert r.status_code == 200
        body = r.json()
        assert isinstance(body, list)
        assert len(body) >= 1
        first = body[0]
        assert "name" in first
        assert "features" in first
        assert "targets" in first

    def test_known_model_present(self, client: TestClient) -> None:
        r = client.get("/v1/models")
        names = {entry["name"] for entry in r.json()}
        assert "TwoTowerGAT" in names


class TestGetModel:
    def test_known_model(self, client: TestClient) -> None:
        r = client.get("/v1/models/TwoTowerGAT")
        assert r.status_code == 200
        body = r.json()
        assert body["name"] == "TwoTowerGAT"
        assert "drugs_cid" in body["features"]
        assert "phenotype_category" in body["targets"]
        assert isinstance(body["fixed_params"], dict)
        assert isinstance(body["optuna_keys"], list)

    def test_unknown_model_404(self, client: TestClient) -> None:
        r = client.get("/v1/models/NotAModel")
        assert r.status_code == 404
        assert "not found" in r.json()["detail"].lower()
