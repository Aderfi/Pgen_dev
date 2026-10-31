"""Shared helpers for the KG tests: a fake ``requests.Session`` over recorded fixtures."""

from __future__ import annotations

import json
import urllib.parse
from pathlib import Path

import pytest
import requests

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "kg"


def fixture_name(key: str) -> str:
    return key.replace(":", "_").replace("/", "_") + ".json"


def _snv_identity(kind: str, expr: str) -> dict | None:
    """An SNV SPDI is its own canonical form (a 1-base substitution cannot shift)."""
    parts = expr.split(":")
    if kind != "canonical_representative" or len(parts) != 4:
        return None
    seq, pos, deleted, inserted = parts
    if len(deleted) != 1 or len(inserted) != 1:
        return None
    return {
        "data": {
            "seq_id": seq,
            "position": int(pos),
            "deleted_sequence": deleted,
            "inserted_sequence": inserted,
        }
    }


class FakeResponse:
    def __init__(
        self, status_code: int, payload: object = None, content: bytes = b""
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.content = content

    def json(self) -> object:
        return self._payload


class FakeSession:
    """Serves recorded NCBI responses; counts calls; can fail on demand.

    ESummary requests for ids not in the fixture get a synthetic minimal
    record, so batching can be exercised with arbitrary id ranges.
    """

    def __init__(
        self, *, fail_after: int | None = None, status_seq: list[int] | None = None
    ) -> None:
        self.calls = 0
        self.urls: list[str] = []
        self.fail_after = fail_after
        self.status_seq = list(status_seq or [])
        self._esummary = json.loads((FIXTURES / "esummary_snp.json").read_text())[
            "result"
        ]

    def get(self, url: str, timeout: float) -> FakeResponse:  # noqa: ARG002, PLR0911
        self.calls += 1
        self.urls.append(url)
        if self.fail_after is not None and self.calls > self.fail_after:
            raise requests.ConnectionError("simulated outage")
        if self.status_seq:
            status = self.status_seq.pop(0)
            if status != 200:
                return FakeResponse(status, {"error": "rate"})
        parsed = urllib.parse.urlparse(url)
        if parsed.path.endswith("einfo.fcgi"):
            return FakeResponse(
                200,
                {
                    "einforesult": {
                        "dbinfo": [
                            {
                                "dbbuild": "Build250306-1408.1",
                                "lastupdate": "2025/03/11 08:22",
                            }
                        ]
                    }
                },
            )
        if parsed.path.endswith("esummary.fcgi"):
            ids = urllib.parse.parse_qs(parsed.query)["id"][0].split(",")
            result: dict[str, object] = {"uids": ids}
            for i in ids:
                result[i] = self._esummary.get(
                    i, {"uid": i, "snp_id": int(i), "spdi": ""}
                )
            return FakeResponse(200, {"result": result})
        path = urllib.parse.unquote(parsed.path)
        for kind in ("canonical_representative", "hgvs", "contextuals"):
            if path.endswith("/" + kind):
                body = path.split("/variation/v0/")[1].rsplit("/", 1)[0]
                expr = body.split("/", 1)[1]
                folder = {
                    "canonical_representative": "spdi_canonical",
                    "hgvs": "spdi_hgvs",
                    "contextuals": "hgvs_contextuals",
                }[kind]
                file = FIXTURES / folder / fixture_name(expr)
                snv = _snv_identity(kind, expr)
                if not file.exists() and snv is not None:
                    return FakeResponse(200, snv)
                if not file.exists():
                    return FakeResponse(
                        400, {"error": {"message": f"no fixture {file.name}"}}
                    )
                return FakeResponse(200, json.loads(file.read_text()))
        return FakeResponse(404, {})


@pytest.fixture
def fake_session() -> FakeSession:
    return FakeSession()
