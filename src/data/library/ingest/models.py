"""Shared ingestion models — the common shape every format adapter produces.

An :class:`IngestedVariant` is one normalised variant: its canonical genomic HGVS
key plus the raw genomic anchor (accession/pos/ref/alt) the graph builder needs,
and optional coding/protein HGVS and Sequence-Ontology consequence terms when the
source supplies them. An :class:`IngestedHaplotype` groups the co-occurring
variants of a named allele (e.g. a PharmVar star allele) — the "star allele =
group of HGVS polymorphisms" unit that becomes a path in the gene graph.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class IngestedVariant(BaseModel):
    """One variant normalised to a canonical genomic HGVS key."""

    model_config = ConfigDict(frozen=True)

    gene: str | None = Field(
        default=None, description="HGNC symbol when known (None for gene-free input)."
    )
    g_hgvs: str = Field(
        ..., description="Canonical genomic HGVS, e.g. 'NC_000022.11:g.42128945C>T'."
    )
    accession: str = Field(
        ..., description="RefSeq genomic accession (NC_*) — the g. reference."
    )
    pos: int = Field(
        ..., ge=1, description="1-based genomic start of the (trimmed) variant."
    )
    ref: str = Field(..., description="Reference allele (VCF-style, uppercased).")
    alt: str = Field(..., description="Alternate allele (VCF-style, uppercased).")
    c_hgvs: str | None = Field(
        default=None, description="Coding HGVS (NM_*:c.*) when a transcript maps."
    )
    p_hgvs: str | None = Field(
        default=None, description="Protein HGVS (NP_*:p.*) when available."
    )
    so_terms: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Sequence Ontology consequence terms (VEP/SnpEff) when annotated.",
    )

    @field_validator("ref", "alt", mode="before")
    @classmethod
    def _upper(cls, v: object) -> str:
        return str(v).strip().upper()


class IngestedHaplotype(BaseModel):
    """A named allele = its ordered group of co-occurring variants."""

    model_config = ConfigDict(frozen=True)

    gene: str = Field(..., description="HGNC symbol the haplotype belongs to.")
    label: str = Field(
        ..., description="Allele label, e.g. '*4' (path name in the gene graph)."
    )
    variants: tuple[IngestedVariant, ...] = Field(
        default_factory=tuple,
        description="The co-occurring variants defining the haplotype (may be empty = reference).",
    )


__all__ = ["IngestedHaplotype", "IngestedVariant"]
