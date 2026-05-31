"""Tests for /v1/predict.

Predictions can't actually run without trained model artifacts, so these
tests focus on:
- request validation (rejecting bad input shapes)
- the 503 path when artifacts are missing
"""

from fastapi.testclient import TestClient


class TestSinglePredict:
    def test_503_when_model_artifacts_missing(self, client: TestClient) -> None:
        body = {
            "drug_cid": 2244,
            "allele": {"gene": {"symbol": "CYP2D6"}, "allele": "4"},
        }
        r = client.post("/v1/predict", json=body)
        # Either 503 (model artifacts missing — expected in dev) OR 200 (artifacts
        # actually present and prediction succeeded).
        assert r.status_code in {200, 503}
        if r.status_code == 503:
            assert (
                "not available" in r.json()["detail"].lower()
                or "failed" in r.json()["detail"].lower()
            )

    def test_rejects_missing_drug_cid(self, client: TestClient) -> None:
        body = {"allele": {"gene": {"symbol": "CYP2D6"}, "allele": "4"}}
        r = client.post("/v1/predict", json=body)
        assert r.status_code == 422

    def test_rejects_missing_allele(self, client: TestClient) -> None:
        body = {"drug_cid": 2244}
        r = client.post("/v1/predict", json=body)
        assert r.status_code == 422

    def test_rejects_negative_cid(self, client: TestClient) -> None:
        body = {"drug_cid": -1, "allele": {"gene": {"symbol": "CYP2D6"}, "allele": "4"}}
        r = client.post("/v1/predict", json=body)
        assert r.status_code == 422

    def test_rejects_extra_fields(self, client: TestClient) -> None:
        body = {
            "drug_cid": 2244,
            "allele": {"gene": {"symbol": "CYP2D6"}, "allele": "4"},
            "rogue": "value",
        }
        r = client.post("/v1/predict", json=body)
        assert r.status_code == 422


class TestBatchPredict:
    def test_rejects_empty_batch(self, client: TestClient) -> None:
        r = client.post("/v1/predict/batch", json={"pairs": []})
        assert r.status_code == 422

    def test_rejects_oversized_batch(self, client: TestClient) -> None:
        # Schema caps at 100 — 101 must fail validation.
        pair = {"drug_cid": 1, "allele": {"gene": {"symbol": "CYP2D6"}, "allele": "1"}}
        r = client.post("/v1/predict/batch", json={"pairs": [pair] * 101})
        assert r.status_code == 422

    def test_503_or_200_with_valid_pairs(self, client: TestClient) -> None:
        pair = {
            "drug_cid": 2244,
            "allele": {"gene": {"symbol": "CYP2D6"}, "allele": "4"},
        }
        r = client.post("/v1/predict/batch", json={"pairs": [pair]})
        assert r.status_code in {200, 503}
