"""Build manifest — track per-CID and per-(gene, variant) progress so the
builder can be interrupted and resumed without losing work.

The manifest is JSON, written atomically (write-temp-then-rename) after each
batch so a crash leaves a consistent file.
"""

from __future__ import annotations

import json
import logging
import tempfile
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


logger = logging.getLogger(__name__)


class ManifestStats(BaseModel):
    drugs_built: int = 0
    drugs_failed: int = 0
    drugs_skipped: int = 0
    genes_built: int = 0
    genes_failed: int = 0
    variants_total: int = 0


class BuildManifest(BaseModel):
    """Persistent record of what's been produced so far."""

    model_config = ConfigDict()

    started_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    completed_drug_cids: list[int] = Field(default_factory=list)
    completed_gene_keys: list[str] = Field(
        default_factory=list,
        description="Strings of the form 'GENE/<variant_name>' for variant graphs already built.",
    )
    failed: dict[str, str] = Field(
        default_factory=dict,
        description="Map of failed key → reason. Drug keys are 'drug:<cid>'; gene keys are 'gene:GENE/<variant>'.",
    )
    stats: ManifestStats = Field(default_factory=ManifestStats)

    @classmethod
    def load_or_empty(cls, path: Path) -> BuildManifest:
        """Read an existing manifest or return a fresh one if absent/corrupt."""
        if not path.exists():
            return cls()
        try:
            with path.open(encoding="utf-8") as f:
                return cls.model_validate(json.load(f))
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("Manifest at %s is unreadable (%s); starting fresh.", path, e)
            return cls()

    def save(self, path: Path) -> None:
        """Atomic write: temp file then rename."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = datetime.now(tz=timezone.utc)
        payload = self.model_dump_json(indent=2)
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, delete=False, prefix=".manifest.", suffix=".tmp"
        ) as tf:
            tf.write(payload)
            tmp_path = Path(tf.name)
        tmp_path.replace(path)

    def mark_drug_done(self, cid: int) -> None:
        if cid not in self.completed_drug_cids:
            self.completed_drug_cids.append(cid)

    def mark_gene_done(self, gene: str, variant: str) -> None:
        key = f"{gene}/{variant}"
        if key not in self.completed_gene_keys:
            self.completed_gene_keys.append(key)

    def mark_drug_failed(self, cid: int, reason: str) -> None:
        self.failed[f"drug:{cid}"] = reason

    def mark_gene_failed(self, gene: str, variant: str, reason: str) -> None:
        self.failed[f"gene:{gene}/{variant}"] = reason

    def has_drug(self, cid: int) -> bool:
        return cid in self.completed_drug_cids

    def has_gene(self, gene: str, variant: str) -> bool:
        return f"{gene}/{variant}" in self.completed_gene_keys

    @staticmethod
    def gene_key_iter(genes_root: Path) -> Iterable[str]:
        """Walk an existing library and yield 'GENE/variant' keys.

        Useful for bootstrapping a manifest on a partially-populated library
        that predates manifest tracking.
        """
        if not genes_root.exists():
            return
        for gene_dir in genes_root.iterdir():
            if not gene_dir.is_dir():
                continue
            for pt in gene_dir.glob("*.pt"):
                stem = pt.stem
                if "_" in stem:
                    _, variant = stem.split("_", 1)
                    yield f"{gene_dir.name}/{variant}"
