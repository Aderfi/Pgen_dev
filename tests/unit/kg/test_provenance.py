"""Tests for provenance primitives and NCBI settings."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from src.config import get_settings
from src.kg.provenance import Dataset, IngestRun, Stamp, code_version, sha256_file


def test_stamp_requires_a_source():
    with pytest.raises(ValidationError):
        Stamp(sources=(), run_id="r")


def test_stamp_props_flatten_for_cypher():
    s = Stamp(sources=("a", "b"), run_id="r1", method="m", evidence="e")
    assert s.props() == {
        "sources": ["a", "b"],
        "run_id": "r1",
        "method": "m",
        "evidence": "e",
    }
    assert Stamp(sources=("a",), run_id="r").props() == {
        "sources": ["a"],
        "run_id": "r",
    }


def test_sha256_file(tmp_path):
    p = tmp_path / "f"
    p.write_bytes(b"abc")
    assert (
        sha256_file(p)
        == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_code_version_outside_git(tmp_path):
    assert code_version(tmp_path) == ("unknown", True)


def test_code_version_in_repo():
    sha, _dirty = code_version(get_settings().paths.project_root)
    assert len(sha) == 40


def test_ingest_run_start():
    run = IngestRun.start({"reset": "true"})
    assert len(run.id) == 32
    assert run.finished_at is None
    assert run.params == {"reset": "true"}


def test_dataset_is_frozen():
    d = Dataset(
        id="x",
        name="x",
        version="1",
        uri="u",
        retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
        sha256="0" * 64,
        license=None,
    )
    with pytest.raises(ValidationError):
        d.id = "y"


def test_ncbi_settings_from_env(monkeypatch):
    monkeypatch.setenv("PHARMAGEN_NCBI__API_KEY", "k")
    get_settings.cache_clear()
    try:
        s = get_settings().ncbi
        assert s.api_key.get_secret_value() == "k"
        assert s.tool == "pharmagen"
        assert s.email is None
    finally:
        get_settings.cache_clear()
