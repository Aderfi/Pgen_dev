"""Request / response models specific to the HTTP layer.

These complement the core domain models (in src.domain.*) — anything that's
purely HTTP envelope (pagination, health probes, single-pair predictions)
lives here.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt

from src.domain.gene import StarAllele
from src.domain.prediction import PredictionResult


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


class SinglePredictRequest(BaseModel):
    """One (drug, allele) pair — the lowest-level prediction request."""

    model_config = ConfigDict(extra="forbid")

    drug_cid: PositiveInt = Field(..., description="PubChem Compound ID.")
    allele: StarAllele


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
