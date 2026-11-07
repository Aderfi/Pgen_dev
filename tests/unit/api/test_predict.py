"""Tests for /v1/predict.

Predictions can't actually run without trained model artifacts, so these
tests focus on:
- request validation (rejecting bad input shapes)
- the 503 path when artifacts are missing
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


class TestSinglePredict:
    def test_503_when_model_artifacts_missing(self, client: TestClient) -> None:
        body = {
            "drug_chembl_id": "CHEMBL25",
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

    def test_rejects_missing_drug(self, client: TestClient) -> None:
        body = {"allele": {"gene": {"symbol": "CYP2D6"}, "allele": "4"}}
        r = client.post("/v1/predict", json=body)
        assert r.status_code == 422

    def test_rejects_missing_allele(self, client: TestClient) -> None:
        body = {"drug_chembl_id": "CHEMBL25"}
        r = client.post("/v1/predict", json=body)
        assert r.status_code == 422

    def test_rejects_malformed_chembl_id(self, client: TestClient) -> None:
        allele = {"gene": {"symbol": "CYP2D6"}, "allele": "4"}
        body = {"drug_chembl_id": "2244", "allele": allele}
        r = client.post("/v1/predict", json=body)
        assert r.status_code == 422

    def test_rejects_extra_fields(self, client: TestClient) -> None:
        body = {
            "drug_chembl_id": "CHEMBL25",
            "allele": {"gene": {"symbol": "CYP2D6"}, "allele": "4"},
            "rogue": "value",
        }
        r = client.post("/v1/predict", json=body)
        assert r.status_code == 422


class TestHgvsPredict:
    def test_accepts_hgvs_genotype(self, client: TestClient) -> None:
        body = {
            "drug_chembl_id": "CHEMBL25",
            "gene": "CYP2C19",
            "hgvs": ["NC_000010.11:g.94781859G>A"],
            "alleles": "AG",
        }
        assert client.post("/v1/predict", json=body).status_code in {200, 503}

    def test_rejects_invalid_hgvs(self, client: TestClient) -> None:
        body = {
            "drug_chembl_id": "CHEMBL25",
            "gene": "CYP2C19",
            "hgvs": ["NC_000010.11:g.?xyz"],
        }
        assert client.post("/v1/predict", json=body).status_code == 422

    def test_rejects_hgvs_without_reference(self, client: TestClient) -> None:
        body = {"drug_chembl_id": "CHEMBL25", "gene": "CYP2C19", "hgvs": ["c.681G>A"]}
        assert client.post("/v1/predict", json=body).status_code == 422


class TestBatchPredict:
    def test_rejects_empty_batch(self, client: TestClient) -> None:
        r = client.post("/v1/predict/batch", json={"pairs": []})
        assert r.status_code == 422

    def test_rejects_oversized_batch(self, client: TestClient) -> None:
        # Schema caps at 100 — 101 must fail validation.
        pair = {
            "drug_chembl_id": "CHEMBL1",
            "allele": {"gene": {"symbol": "CYP2D6"}, "allele": "1"},
        }
        r = client.post("/v1/predict/batch", json={"pairs": [pair] * 101})
        assert r.status_code == 422

    def test_503_or_200_with_valid_pairs(self, client: TestClient) -> None:
        pair = {
            "drug_chembl_id": "CHEMBL25",
            "allele": {"gene": {"symbol": "CYP2D6"}, "allele": "4"},
        }
        r = client.post("/v1/predict/batch", json={"pairs": [pair]})
        assert r.status_code in {200, 503}
