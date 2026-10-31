"""Append-only JSONL cache of raw upstream responses.

Every external fact the KG uses is first written here, verbatim, with the
exact request and retrieval time. The graph is then built from this cache
alone, so a build is reproducible offline and every value can be traced back
to the response it came from.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from src.core import DataError

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class CacheEntry(BaseModel):
    """One upstream response."""

    key: str
    request: str = Field(..., description="Request URL, never including secrets.")
    status: Literal["ok", "not_found", "error"]
    retrieved_at: datetime
    response: dict | list | None = None
    warnings: list[str] = Field(default_factory=list)
    source_version: str | None = Field(
        default=None, description="Upstream release, e.g. the dbSNP build."
    )


class CacheMissError(DataError):
    """Raised in offline mode when required responses were never fetched."""

    def __init__(self, missing: list[str]) -> None:
        self.missing = missing
        preview = ", ".join(missing[:10])
        super().__init__(
            f"{len(missing)} required upstream responses are not cached "
            f"(first: {preview}); run 'python -m src.kg fetch'"
        )


class RawCache:
    """Key → :class:`CacheEntry` map persisted as one JSON object per line.

    The file is append-only: a refreshed response is appended and the last
    line for a key wins on reload, so the full retrieval history is kept.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._entries: dict[str, CacheEntry] = {}
        self._needs_newline = False
        if path.exists():
            self._load()

    def _load(self) -> None:
        text = self.path.read_text()
        lines = text.splitlines()
        last = max((n for n, line in enumerate(lines) if line.strip()), default=-1)
        for n, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                entry = CacheEntry.model_validate_json(line)
            except ValueError as e:
                if n == last:
                    # A process killed mid-write leaves a partial final line.
                    logger.warning(
                        "%s: skipping truncated final line %d", self.path, n + 1
                    )
                    continue
                msg = f"{self.path}:{n + 1}: corrupt cache line: {e}"
                raise DataError(msg) from e
            self._entries[entry.key] = entry  # last write wins
        self._needs_newline = bool(text) and not text.endswith("\n")

    def get(self, key: str) -> CacheEntry | None:
        return self._entries.get(key)

    def put(self, key: str, entry: CacheEntry, *, overwrite: bool = False) -> None:
        """Append ``entry``; an existing key is replaced only with ``overwrite``."""
        if key in self._entries and not overwrite:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(entry.model_dump(mode="json"), sort_keys=True) + "\n"
        with self.path.open("a") as f:
            f.write(("\n" if self._needs_newline else "") + line)
        self._needs_newline = False
        self._entries[key] = entry

    def keys(self) -> set[str]:
        return set(self._entries)

    def entries(self) -> list[CacheEntry]:
        return list(self._entries.values())

    def sha256(self) -> str:
        """Checksum of the cache file as it is on disk (empty file if absent)."""
        data = self.path.read_bytes() if self.path.exists() else b""
        return hashlib.sha256(data).hexdigest()


__all__ = ["CacheEntry", "CacheMissError", "RawCache"]
