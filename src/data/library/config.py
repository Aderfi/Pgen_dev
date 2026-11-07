"""Pydantic configuration for a library build run."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from src.config import ModephereSettings, get_settings
from src.data.genotype.catalog import HGVS_CATALOG_FILE

ADMET_FILE = "admet_profile.parquet"


class LibraryBuildConfig(BaseModel):
    """What one ``python -m src.data.library`` run reads, writes and does.

    Modephere's locations and Neo4j connection travel in ``modephere``
    (``Settings.modephere``); dev_Pgen's own outputs — the ADMET table and the
    HGVS catalog — live under ``library_root`` (``Settings.paths.library``).
    """

    model_config = ConfigDict(frozen=True)

    modephere: ModephereSettings
    library_root: Path = Field(
        ..., description="dev_Pgen's library root (hgvs_catalog.pt, ADMET table)."
    )
    star_alleles_tsv: Path = Field(
        ...,
        description="Star-allele function table (gene, allele, function, notes) "
        "— the PGx function of star alleles in the genotype graphs.",
    )

    refresh_pgx: bool = Field(
        default=False,
        description="Re-export Modephere's pgx tables from its Neo4j first.",
    )
    force: bool = Field(default=False, description="Rebuild drugs already stored.")
    skip_drugs: bool = False
    skip_genes: bool = False
    skip_admet: bool = Field(
        default=False,
        description="Leave the ADMET table as it is (drugs without a profile get "
        "a zero admet_feats; no GPU / fast builds).",
    )
    force_admet: bool = Field(
        default=False, description="Recompute every ADMET profile."
    )

    @property
    def hgvs_catalog_path(self) -> Path:
        return self.library_root / HGVS_CATALOG_FILE

    @property
    def admet_path(self) -> Path:
        return self.library_root / ADMET_FILE

    @classmethod
    def from_settings(cls, **flags: bool) -> LibraryBuildConfig:
        """Config rooted at the project's settings; ``flags`` set the behaviour."""
        settings = get_settings()
        return cls(
            modephere=settings.modephere,
            library_root=settings.paths.library,
            star_alleles_tsv=settings.paths.data / "dicts" / "star_alleles.tsv",
            **flags,
        )


__all__ = ["ADMET_FILE", "LibraryBuildConfig"]
