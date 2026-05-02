"""Prediction request / response models — the contract for FastAPI inference.

Inputs reference drugs by PubChem CID (most stable cross-DB identifier) and
genotype as a list of star alleles. Outputs are per-target dictionaries with
predicted label + calibrated probability.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.domain.gene import StarAllele


class PredictionRequest(BaseModel):
    """Single-patient prediction request.

    `drugs` is a list of PubChem CIDs — names are not accepted at the API
    boundary because they are ambiguous (multiple synonyms map to the same CID,
    and resolution depends on the local PubChem snapshot).
    """

    model_config = ConfigDict(extra="forbid")

    drugs: list[int] = Field(..., min_length=1, description="PubChem CIDs of prescribed drugs.")
    genotype: list[StarAllele] = Field(..., description="Patient star-allele profile.")
    sample_id: str | None = None


class TargetPrediction(BaseModel):
    """Prediction for a single target column."""

    model_config = ConfigDict(frozen=True)

    target: str = Field(..., description="Target column name (e.g. 'phenotype_category').")
    label: str = Field(..., description="Predicted class label.")
    probability: float = Field(..., ge=0.0, le=1.0)
    probabilities: dict[str, float] = Field(
        default_factory=dict,
        description="Full per-class probability distribution.",
    )


class PredictionResult(BaseModel):
    """Top-level prediction response covering all model targets."""

    model_config = ConfigDict(frozen=True)

    request_id: str | None = None
    model_name: str
    model_version: str | None = None
    predictions: list[TargetPrediction]
    generated_at: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)
