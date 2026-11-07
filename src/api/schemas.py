"""Request / response models specific to the HTTP layer.

These complement the core domain models (in src.domain.*) — anything that's
purely HTTP envelope (pagination, health probes, single-pair predictions)
lives here.
"""

from __future__ import annotations

import re
from typing import Any

from modephere.types import ChemblID
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    field_validator,
    model_validator,
)

from src.core import BioinformaticsError
from src.domain.schemas.gene import StarAllele
from src.domain.schemas.prediction import PredictionResult
from src.genomics.hgvs_parser import parse as parse_hgvs


class HealthResponse(BaseModel):
    status: str
    version: str
    project: str


class ReadinessResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str | None = None
    detail: str | None = None


class ModelSummary(BaseModel):
    name: str
    features: list[str]
    targets: list[str]


class ModelDetail(ModelSummary):
    cols: list[str]
    stratify_col: str | None = None
    fixed_params: dict[str, Any] = Field(default_factory=dict)
    optuna_keys: list[str] = Field(default_factory=list)
    extras: dict[str, Any] = Field(default_factory=dict)


#: A drug entity key: one drug (ChEMBL ID) or an ATC class (codes, comma-separated).
_ENTITY_KEY = re.compile(r"CHEMBL\d+|ATC:[A-Z0-9]+(,[A-Z0-9]+)*")


class SinglePredictRequest(BaseModel):
    """One drug + one genotype — the lowest-level prediction request.

    The genotype is stated as HGVS variants (``hgvs``), optionally with a star
    allele (``allele``, expanded to its HGVS variants by the catalog) and the
    observed ``alleles`` (zygosity, e.g. ``"CT"`` or ``"*1/*4"``). A gene with
    no variants is the reference genotype.
    """

    model_config = ConfigDict(extra="forbid")

    drug_chembl_id: ChemblID | None = Field(
        default=None, description="ChEMBL parent ID of the drug, e.g. 'CHEMBL1771'."
    )
    drug_ids: list[str] | None = Field(
        default=None,
        min_length=1,
        max_length=20,
        description="The drugs taken together: ChEMBL parent IDs and/or ATC "
        "classes ('ATC:N02A'), e.g. ['CHEMBL443', 'CHEMBL22']. Give this or "
        "'drug_chembl_id'.",
    )
    gene: str | None = Field(
        default=None,
        min_length=1,
        description="HGNC symbol (taken from `allele` if omitted).",
    )
    hgvs: list[str] = Field(
        default_factory=list,
        max_length=50,
        description="HGVS expressions with reference, e.g. 'NC_000010.11:g.94781859G>A'.",
    )
    allele: StarAllele | None = None
    alleles: str | None = Field(
        default=None, description="Observed genotype / diplotype, e.g. 'CT' or '*1/*4'."
    )

    @field_validator("hgvs")
    @classmethod
    def _valid_hgvs(cls, values: list[str]) -> list[str]:
        for value in values:
            if ":" not in value:
                raise ValueError(f"HGVS {value!r} needs a reference sequence prefix")
            try:
                parse_hgvs(value)
            except BioinformaticsError as e:
                raise ValueError(f"invalid HGVS {value!r}: {e}") from e
        return values

    @field_validator("drug_ids")
    @classmethod
    def _entity_keys(cls, values: list[str] | None) -> list[str] | None:
        if values is None:
            return None
        keys = [v.strip().upper() for v in values]
        bad = [k for k in keys if not _ENTITY_KEY.fullmatch(k)]
        if bad:
            raise ValueError(
                f"not a ChEMBL ID or ATC class key (CHEMBL<n> / ATC:<code>): {bad}"
            )
        return keys

    @model_validator(mode="after")
    def _one_drug_field(self) -> SinglePredictRequest:
        if (self.drug_chembl_id is None) == (self.drug_ids is None):
            raise ValueError("give exactly one of 'drug_chembl_id' or 'drug_ids'")
        return self

    @property
    def drug_entities(self) -> list[str]:
        """The request's drug entity keys (upper case)."""
        if self.drug_ids is not None:
            return self.drug_ids
        return [str(self.drug_chembl_id).upper()]

    @model_validator(mode="after")
    def _needs_gene(self) -> SinglePredictRequest:
        if self.gene is None and self.allele is None:
            raise ValueError("give 'gene' (with 'hgvs') or a star 'allele'")
        return self

    @property
    def gene_symbol(self) -> str:
        return self.gene or self.allele.gene.symbol  # type: ignore[union-attr]


class BatchPredictRequest(BaseModel):
    """Multiple pairs predicted in one call. Capped to keep request size sane."""

    model_config = ConfigDict(extra="forbid")

    pairs: list[SinglePredictRequest] = Field(..., min_length=1, max_length=100)


class BatchPredictResponse(BaseModel):
    results: list[PredictionResult]
    count: int


class LibraryEntry(BaseModel):
    kind: str
    identifier: str
    path: str


class LibraryListResponse(BaseModel):
    kind: str
    total: NonNegativeInt
    offset: NonNegativeInt
    limit: PositiveInt
    items: list[LibraryEntry]
