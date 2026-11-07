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
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any

from modephere.config import KGSettings as ModephereKGSettings
from pydantic import BaseModel, Field, PositiveInt, SecretStr, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from src.config.paths import Paths
from src.domain.schemas.variant import GenomeBuild

# Resolved at import time once — the project root is derived from this file's
# location (3 levels up: src/config/settings.py → src/config → src → root).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
CONFIG_DATA_DIR: Path = Path(__file__).parent / "data"


class KGSettings(BaseModel):
    """Neo4j knowledge-graph connection.

    Override via env, e.g. ``PHARMAGEN_KG__URI=bolt://host:7687`` and
    ``PHARMAGEN_KG__PASSWORD=...``. The password never lives in TOML.
    """

    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: SecretStr = SecretStr("neo4j")
    database: str = "neo4j"


class NCBISettings(BaseModel):
    """NCBI E-utilities / Variation Services client identity.

    ``PHARMAGEN_NCBI__API_KEY`` raises the E-utilities limit from 3 to 10
    requests/s; ``PHARMAGEN_NCBI__EMAIL`` is sent only when set.
    """

    api_key: SecretStr | None = None
    email: str | None = None
    tool: str = "pharmagen"


class ModephereSettings(BaseModel):
    """Where Modephere reads and writes when dev_Pgen drives it.

    Everything Modephere touches for dev_Pgen comes from here and is passed to
    it explicitly (``src.data.library.bridge``); Modephere's own settings,
    ``MODEPHERE_*`` variables and working directory are never consulted.
    Relative paths resolve against the project root. Override with
    ``PHARMAGEN_MODEPHERE__<FIELD>``, e.g. ``PHARMAGEN_MODEPHERE__LIBRARY``,
    ``PHARMAGEN_MODEPHERE__KG__DATABASE`` or ``PHARMAGEN_MODEPHERE__KG__PASSWORD``.
    """

    library: Path = Field(
        default=Path("data/library"),
        description="Modephere LibraryStore root: compounds/<InChIKey>.safetensors "
        "plus compounds / conformers / failures parquet.",
    )
    cache: Path = Field(
        default=Path("data/cache/modephere"),
        description="Modephere's raw PubChem cache (drugs ChEMBL cannot resolve).",
    )
    chembl: Path = Field(
        default=Path("data/chembl"),
        description="Staged ChEMBL releases (chembl_<N>/); the newest resolves "
        "ChEMBL IDs offline.",
    )
    pgx_export: Path = Field(
        default=Path("data/library/pgx"),
        description="Modephere pgx export tables (annotation_rows, allele_variants, "
        "annotation_variants, annotation_genotypes, variants).",
    )
    logs: Path = Field(
        default=Path("logs/library"),
        description="Library build failure / feature-saturation reports.",
    )
    max_heavy_atoms: PositiveInt = Field(
        default=300,
        description="Heavy-atom cap of the standardizer (peptides such as "
        "exenatide, 295, fit under 300).",
    )
    kg: ModephereKGSettings = Field(
        default_factory=ModephereKGSettings,
        description="Modephere's Neo4j (pgx layer), read by 'library --refresh-pgx'.",
    )

    @field_validator("library", "cache", "chembl", "pgx_export", "logs")
    @classmethod
    def _absolute(cls, v: Path) -> Path:
        v = v.expanduser()
        return v if v.is_absolute() else PROJECT_ROOT / v


class Settings(BaseSettings):
    """Top-level project settings.

    Sourced (in priority order) from:
      1. Constructor arguments / explicit overrides
      2. Environment variables (PHARMAGEN_* prefix)
      3. ``src/config/data/{paths,settings}.toml``
      4. Field defaults

    ``[modephere]`` in ``settings.toml`` is read as a settings source, below the
    environment, so ``PHARMAGEN_MODEPHERE__*`` variables win over it.
    """

    model_config = SettingsConfigDict(
        env_prefix="PHARMAGEN_",
        env_nested_delimiter="__",
        extra="ignore",
        toml_file=CONFIG_DATA_DIR / "settings.toml",
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

    # Knowledge graph (Neo4j)
    kg: KGSettings = Field(default_factory=KGSettings)
    ncbi: NCBISettings = Field(default_factory=NCBISettings)

    # Modephere (drug library + pgx exports) — see ModephereSettings
    modephere: ModephereSettings = Field(default_factory=ModephereSettings)

    # Filesystem
    paths: Paths

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (init_settings, env_settings, TomlConfigSettingsSource(settings_cls))

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
        models=_abs(models.get("models_saved", "data/pgen_model/models")),
        encoders=_abs(models.get("encoders", "data/pgen_model/encoders")),
        ref_genome_dir=_abs(genome.get("ref_genome_dir", "data/ref_genome")),
        ref_genome_fasta=_abs(
            genome.get("ref_genome_fasta", "data/ref_genome/HSapiens_GChr38.fa")
        ),
        ref_genome_gff=_abs(
            genome.get("ref_genome_gff", "data/ref_genome/HSapiens_GChr38.gff")
        ),
        ref_genome_rna_fasta=_abs(
            genome.get(
                "ref_genome_rna_fasta", "data/ref_genome/HSapiens_GChr38.rna.fna"
            )
        ),
        ref_genome_cds_fasta=_abs(
            genome.get(
                "ref_genome_cds_fasta", "data/ref_genome/HSapiens_GChr38.cds.fna"
            )
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
