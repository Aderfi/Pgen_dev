"""Tests for /health and /ready."""

from fastapi.testclient import TestClient


class TestHealth:
    def test_returns_200(self, client: TestClient) -> None:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["project"] == "Pharmagen"
        assert "version" in body


class TestReady:
    def test_returns_200_even_without_model(self, client: TestClient) -> None:
        # /ready never errors — it reports the loaded state, including pending.
        r = client.get("/ready")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] in {"ready", "pending"}
        assert isinstance(body["model_loaded"], bool)
        assert "model_name" in body
