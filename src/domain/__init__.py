"""Pharmagen domain models — Pydantic v2.

The single source of truth for the project's bio + ML data shapes.
Every public boundary (FastAPI, CLI, training pipeline, predictor) accepts and
returns these models; never raw dicts or unvalidated strings.
"""

from src.domain.drug import Drug
from src.domain.gene import AlleleFunction, Gene, StarAllele
from src.domain.graph import GraphKind, GraphMetadata, GraphPair
from src.domain.prediction import (
    PredictionRequest,
    PredictionResult,
    TargetPrediction,
)
from src.domain.variant import (
    GenomeBuild,
    Genotype,
    Position,
    Variant,
    VariantType,
    Zygosity,
)

__all__ = [
    "AlleleFunction",
    "Drug",
    "Gene",
    "GenomeBuild",
    "Genotype",
    "GraphKind",
    "GraphMetadata",
    "GraphPair",
    "PredictionRequest",
    "PredictionResult",
    "Position",
    "StarAllele",
    "TargetPrediction",
    "Variant",
    "VariantType",
    "Zygosity",
]
