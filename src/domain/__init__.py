"""Pharmagen domain models — Pydantic v2.

The single source of truth for the project's bio + ML data shapes.
Every public boundary (FastAPI, CLI, training pipeline, predictor) accepts and
returns these models; never raw dicts or unvalidated strings.
"""

from src.domain.dbsnp import (
    DbSnpGene,
    DbSnpSummary,
    SpdiAllele,
    build_from_accession,
)
from src.domain.drug import Drug
from src.domain.gene import AlleleFunction, Gene, StarAllele
from src.domain.graph import GraphKind, GraphMetadata, GraphPair
from src.domain.hgvs import (
    HGVSChange,
    HGVSVariant,
    MolecularType,
    NucleotideChange,
    ProteinChange,
    ProteinPosition,
    ReferenceSequenceKind,
    SequencePosition,
    VariantKind,
    VariantPhase,
)
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
    "DbSnpGene",
    "DbSnpSummary",
    "Drug",
    "Gene",
    "GenomeBuild",
    "Genotype",
    "GraphKind",
    "GraphMetadata",
    "GraphPair",
    "HGVSChange",
    "HGVSVariant",
    "MolecularType",
    "NucleotideChange",
    "PredictionRequest",
    "PredictionResult",
    "Position",
    "ProteinChange",
    "ProteinPosition",
    "ReferenceSequenceKind",
    "SequencePosition",
    "SpdiAllele",
    "StarAllele",
    "TargetPrediction",
    "Variant",
    "VariantKind",
    "VariantPhase",
    "VariantType",
    "Zygosity",
    "build_from_accession",
]
