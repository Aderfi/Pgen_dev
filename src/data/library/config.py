"""Pydantic configuration for a library build run."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.config import get_settings


class LibraryBuildConfig(BaseModel):
    """All inputs and outputs for a library build, validated up front.

    Defaults are derived from ``Settings`` so callers can usually do
    ``LibraryBuildConfig.from_settings()`` and be done. Any field can be
    overridden at construction.
    """

    model_config = ConfigDict(frozen=True)

    # Inputs
    variants_tsv: Path = Field(
        ..., description="Pan-gene variant TSV (snp_data_output.tsv style)."
    )
    drugs_tsv: Path = Field(
        ..., description="Drug catalog TSV (cid, smiles, cmpd_name_cleaned)."
    )
    fasta_path: Path = Field(
        ..., description="Reference FASTA (must be indexed; .fai required)."
    )
    pgx_dir: Path = Field(
        ..., description="Per-gene PharmVar VCF folder (one subdir per gene)."
    )

    # Outputs
    library_root: Path = Field(
        ..., description="Root of the produced library (drugs/ and gene_graphs/)."
    )

    # Behaviour
    force: bool = Field(
        default=False, description="Overwrite existing .pt files (default: skip)."
    )
    only_gene: str | None = Field(
        default=None,
        description="If set, build only this gene's variants — useful for verification.",
    )
    skip_drugs: bool = False
    skip_genes: bool = False
    log_failures_path: Path | None = Field(
        default=None,
        description="Optional path for the drug-generation error log; defaults to library_root/build_failures.log.",
    )

    @field_validator(
        "variants_tsv",
        "drugs_tsv",
        "fasta_path",
        "pgx_dir",
        "library_root",
        mode="before",
    )
    @classmethod
    def _coerce_path(cls, v: object) -> Path:
        return Path(v) if not isinstance(v, Path) else v

    @property
    def drugs_out(self) -> Path:
        return self.library_root / "drugs"

    @property
    def genes_out(self) -> Path:
        return self.library_root / "gene_graphs"

    @property
    def manifest_path(self) -> Path:
        return self.library_root / "build_manifest.json"

    def ensure_outputs(self) -> None:
        """Create output directories. Inputs are NOT created — they must exist."""
        self.library_root.mkdir(parents=True, exist_ok=True)
        self.drugs_out.mkdir(parents=True, exist_ok=True)
        self.genes_out.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_settings(
        cls,
        *,
        variants_tsv: Path | None = None,
        drugs_tsv: Path | None = None,
        force: bool = False,
        only_gene: str | None = None,
        skip_drugs: bool = False,
        skip_genes: bool = False,
    ) -> LibraryBuildConfig:
        """Build a config rooted at the project's standard paths."""
        settings = get_settings()
        data_dir = settings.paths.data
        return cls(
            variants_tsv=variants_tsv or data_dir / "snp_data_output.tsv",
            drugs_tsv=drugs_tsv or data_dir / "drugs_cid.tsv",
            fasta_path=settings.paths.ref_genome_fasta,
            pgx_dir=data_dir / "haplotype_variants",
            library_root=settings.paths.library,
            force=force,
            only_gene=only_gene,
            skip_drugs=skip_drugs,
            skip_genes=skip_genes,
        )
