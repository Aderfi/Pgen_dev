"""Project-wide settings — Pydantic Settings backed by TOML + env vars.

Public API:
    >>> from src.config import get_settings
    >>> settings = get_settings()
    >>> settings.paths.data
    PosixPath('/.../data')

Override via environment variables prefixed ``PHARMAGEN_`` (e.g.
``PHARMAGEN_LOG_LEVEL=DEBUG``). Nested fields use double-underscore:
``PHARMAGEN_PATHS__DATA=/tmp/data``.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import tomllib
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.config.paths import Paths
from src.domain.variant import GenomeBuild

# Resolved at import time once — the project root is derived from this file's
# location (3 levels up: src/config/settings.py → src/config → src → root).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
CONFIG_DATA_DIR: Path = Path(__file__).parent / "data"


class Settings(BaseSettings):
    """Top-level project settings.

    Sourced (in priority order) from:
      1. Constructor arguments / explicit overrides
      2. Environment variables (PHARMAGEN_* prefix)
      3. ``src/config/data/{paths,settings}.toml``
      4. Field defaults
    """

    model_config = SettingsConfigDict(
        env_prefix="PHARMAGEN_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Metadata
    project_name: str = "Pharmagen"
    version: str = "0.0.0"
    author: str = ""

    # Behavior
    seed: int = 711
    log_level: str = "WARNING"
    multi_label_cols: list[str] = Field(default_factory=list)
    genome_build: GenomeBuild = GenomeBuild.GRCH38

    # Filesystem
    paths: Paths

    @property
    def multi_label_set(self) -> set[str]:
        """Convenient set view (legacy code uses set semantics)."""
        return set(self.multi_label_cols)


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        msg = f"Configuration file not found: {path}"
        raise FileNotFoundError(msg)
    with path.open("rb") as f:
        return tomllib.load(f)


def _build_paths(paths_cfg: dict[str, Any]) -> Paths:
    """Translate the [base]/[models]/[genome_references] TOML sections into Paths."""
    base = paths_cfg.get("base", {})
    models = paths_cfg.get("models", {})
    genome = paths_cfg.get("genome_references", {})

    def _abs(rel: str) -> Path:
        return PROJECT_ROOT / rel

    return Paths(
        project_root=PROJECT_ROOT,
        data=_abs(base.get("data", "data")),
        logs=_abs(base.get("logs", "logs")),
        results=_abs(base.get("results", "results")),
        reports=_abs(base.get("reports", "reports")),
        cache=_abs(base.get("cache", "cache")),
        library=_abs(base.get("library", "data/library")),
        models=_abs(models.get("models_saved", "src/pgen_model/models")),
        encoders=_abs(models.get("encoders", "src/pgen_model/encoders")),
        ref_genome_dir=_abs(genome.get("ref_genome_dir", "data/ref_genome")),
        ref_genome_fasta=_abs(
            genome.get("ref_genome_fasta", "data/ref_genome/HSapiens_GChr38.fa")
        ),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache the project settings.

    The cache means TOML files are read only once per process. Tests that need
    a fresh load can call ``get_settings.cache_clear()``.
    """
    paths_cfg = _load_toml(CONFIG_DATA_DIR / "paths.toml")
    settings_cfg = _load_toml(CONFIG_DATA_DIR / "settings.toml")

    metadata = settings_cfg.get("metadata", {})
    project = settings_cfg.get("project", {})

    paths = _build_paths(paths_cfg)

    return Settings(
        project_name=metadata.get("project_name", "Pharmagen"),
        version=metadata.get("version", "0.0.0"),
        author=metadata.get("author", ""),
        seed=project.get("seed", 711),
        multi_label_cols=project.get("multi_label_cols", []),
        paths=paths,
    )


def configure_logging_level(settings: Settings | None = None) -> None:
    """Apply the configured log level to the root logger."""
    s = settings or get_settings()
    level = getattr(logging, s.log_level.upper(), logging.WARNING)
    logging.getLogger().setLevel(level)
