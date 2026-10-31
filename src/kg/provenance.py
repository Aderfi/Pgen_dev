"""Provenance primitives, following the W3C PROV-O concepts.

* :class:`Dataset` — a ``prov:Entity``: one pinned input (a file or an online
  source at a given release), with checksum and retrieval time.
* :class:`IngestRun` — a ``prov:Activity``: one execution of the KG build,
  tied to the exact code version that ran.
* :class:`Stamp` — what every node and relationship carries: the datasets it
  derives from, the run that wrote it and, for derived facts, the method and
  the raw evidence.
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

_CHUNK = 1 << 20


class Dataset(BaseModel):
    """A pinned input to the knowledge graph (``prov:Entity``)."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(..., min_length=1)
    name: str
    version: str
    uri: str
    retrieved_at: datetime
    sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    license: str | None = None


class IngestRun(BaseModel):
    """One KG build execution (``prov:Activity``)."""

    id: str
    started_at: datetime
    finished_at: datetime | None = None
    code_version: str
    dirty: bool
    params: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def start(cls, params: dict[str, str], repo: Path | None = None) -> IngestRun:
        from src.config import get_settings

        sha, dirty = code_version(repo or get_settings().paths.project_root)
        return cls(
            id=uuid.uuid4().hex,
            started_at=datetime.now(UTC),
            code_version=sha,
            dirty=dirty,
            params=dict(params),
        )


class Stamp(BaseModel):
    """Provenance carried by every node and relationship."""

    model_config = ConfigDict(frozen=True)

    sources: tuple[str, ...] = Field(..., min_length=1)
    run_id: str = Field(..., min_length=1)
    method: str | None = None
    evidence: str | None = None

    def props(self) -> dict[str, object]:
        """Flat property map for Cypher (``None`` fields omitted)."""
        out: dict[str, object] = {"sources": list(self.sources), "run_id": self.run_id}
        if self.method is not None:
            out["method"] = self.method
        if self.evidence is not None:
            out["evidence"] = self.evidence
        return out


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def code_version(repo: Path) -> tuple[str, bool]:
    """``(commit sha, dirty)`` of ``repo``; ``("unknown", True)`` outside git."""
    try:
        sha = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError) as e:
        logger.warning("Cannot read git version of %s: %s", repo, e)
        return "unknown", True
    return sha, bool(status.strip())


__all__ = ["Dataset", "IngestRun", "Stamp", "code_version", "sha256_file"]
