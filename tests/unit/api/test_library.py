"""Tests for /v1/library/{drugs,genes}."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


class TestLibrary:
    def test_drugs_pagination_envelope(self, client: TestClient) -> None:
        r = client.get("/v1/library/drugs?limit=5")
        assert r.status_code == 200
        body = r.json()
        assert body["kind"] == "drug"
        assert isinstance(body["total"], int)
        assert body["limit"] == 5
        assert body["offset"] == 0
        assert isinstance(body["items"], list)

    def test_genes_pagination_envelope(self, client: TestClient) -> None:
        r = client.get("/v1/library/genes?limit=10&offset=0")
        assert r.status_code == 200
        body = r.json()
        assert body["kind"] == "gene"
        assert body["limit"] == 10

    def test_unknown_gene_404(self, client: TestClient) -> None:
        r = client.get("/v1/library/genes/NOT_A_GENE_XYZ")
        assert r.status_code == 404

    def test_invalid_limit_rejected(self, client: TestClient) -> None:
        # Schema caps limit at 500
        r = client.get("/v1/library/drugs?limit=10000")
        assert r.status_code == 422

    def test_negative_offset_rejected(self, client: TestClient) -> None:
        r = client.get("/v1/library/drugs?offset=-1")
        assert r.status_code == 422
