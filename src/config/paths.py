"""Filesystem paths used by Pharmagen — typed and validated.

A single Path-aware Pydantic model. Directory creation is **not** automatic on
import; callers run ``Settings.paths.ensure_dirs()`` explicitly (typically from
``main.py`` startup or the FastAPI lifespan).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator


class Paths(BaseModel):
    """Resolved filesystem layout for the project.

    All fields are absolute Paths. The ``project_root`` is set by the loader
    and the rest are derived from it before construction.
    """

    model_config = ConfigDict(frozen=True)

    project_root: Path
    data: Path
    logs: Path
    results: Path
    reports: Path
    models: Path
    encoders: Path
    cache: Path
    ref_genome_dir: Path
    ref_genome_fasta: Path

    @field_validator("project_root", mode="before")
    @classmethod
    def _resolve_root(cls, v: Path | str) -> Path:
        return Path(v).resolve()

    def ensure_dirs(self) -> None:
        """Create directories that should always exist. Files (e.g. the FASTA)
        are not created — those are populated by the genome download step."""
        for d in (
            self.data,
            self.logs,
            self.results,
            self.reports,
            self.models,
            self.encoders,
            self.cache,
            self.ref_genome_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)
