"""Pydantic configuration for a library build run."""

from __future__ import annotations

from datetime import UTC, datetime
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
        ...,
        description="Drug catalog: .tsv/.csv (cid, smiles, cmpd_name_cleaned) "
        "or .json ({cid: smiles}).",
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
    skip_admet: bool = Field(
        default=False,
        description="Skip ADMET prediction; drug graphs get a zero admet_feats "
        "vector instead of the predicted profile (no GPU / fast CI builds).",
    )
    force_admet: bool = Field(
        default=False,
        description="Recompute the ADMET cache even if a valid one exists.",
    )
    admet_cache: Path = Field(
        ..., description="Parquet cache for the predicted ADMET profile table."
    )
    skip_geno_func: bool = Field(
        default=False,
        description="Skip the genotype functional profile; variant graphs get a "
        "zero geno_global_feats vector instead of PGx-function + pathogenicity.",
    )
    star_alleles_tsv: Path = Field(
        ...,
        description="Star-allele function table (gene, allele, rsids, function, "
        "notes) — Layer A of geno_global_feats.",
    )
    alphamissense_path: Path | None = Field(
        default=None,
        description="Optional (chrom,pos,ref,alt,alphamissense) table — Layer B "
        "pathogenicity. Absent ⇒ those dims stay zero (mask 0).",
    )
    cadd_path: Path | None = Field(
        default=None,
        description="Optional (chrom,pos,ref,alt,cadd_phred) table — Layer B "
        "pathogenicity. Absent ⇒ those dims stay zero (mask 0).",
    )
    strip_salts: bool = Field(
        default=True,
        description="Reduce multi-fragment drug SMILES to their largest fragment "
        "(drop salt counterions) before building the graph.",
    )
    log_failures_path: Path | None = Field(
        default=None,
        description="Path for the drug-generation failure report; defaults to "
        "logs/library/drug_build_failures_<date>.log.",
    )
    log_saturation_path: Path | None = Field(
        default=None,
        description="Path for the one-hot feature-saturation report; defaults to "
        "logs/library/drug_feature_saturation_<date>.log.",
    )

    @field_validator(
        "variants_tsv",
        "drugs_tsv",
        "fasta_path",
        "pgx_dir",
        "library_root",
        "admet_cache",
        "star_alleles_tsv",
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
        skip_admet: bool = False,
        force_admet: bool = False,
        skip_geno_func: bool = False,
        strip_salts: bool = True,
    ) -> LibraryBuildConfig:
        """Build a config rooted at the project's standard paths."""
        settings = get_settings()
        data_dir = settings.paths.data
        dicts_dir = data_dir / "dicts"
        stamp = datetime.now(tz=UTC).strftime("%Y%m%d")
        library_logs = settings.paths.logs / "library"
        log_failures_path = library_logs / f"drug_build_failures_{stamp}.log"
        log_saturation_path = library_logs / f"drug_feature_saturation_{stamp}.log"
        # Layer B (pathogenicity) auto-enables when the conventional file is present.
        am_default = dicts_dir / "alphamissense_pgx.tsv"
        cadd_default = dicts_dir / "cadd_pgx.tsv"
        return cls(
            variants_tsv=variants_tsv or data_dir / "snp_data_output.tsv",
            drugs_tsv=drugs_tsv or data_dir / "drugs_cid.tsv",
            fasta_path=settings.paths.ref_genome_fasta,
            pgx_dir=data_dir / "haplotype_variants",
            library_root=settings.paths.library,
            admet_cache=settings.paths.library / "admet_profile.parquet",
            star_alleles_tsv=dicts_dir / "star_alleles.tsv",
            alphamissense_path=am_default if am_default.exists() else None,
            cadd_path=cadd_default if cadd_default.exists() else None,
            force=force,
            only_gene=only_gene,
            skip_drugs=skip_drugs,
            skip_genes=skip_genes,
            skip_admet=skip_admet,
            force_admet=force_admet,
            skip_geno_func=skip_geno_func,
            strip_salts=strip_salts,
            log_failures_path=log_failures_path,
            log_saturation_path=log_saturation_path,
        )
