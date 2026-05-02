"""Tests for src.domain.prediction."""

import pytest

from src.domain.gene import AlleleFunction, StarAllele
from src.domain.prediction import (
    PredictionRequest,
    PredictionResult,
    TargetPrediction,
)


class TestPredictionRequest:
    def test_minimal(self) -> None:
        sa = StarAllele.parse("CYP2D6*1")
        req = PredictionRequest(drugs=[2244], genotype=[sa])
        assert req.drugs == [2244]
        assert req.genotype[0].label == "CYP2D6*1"

    def test_empty_drugs_rejected(self) -> None:
        with pytest.raises(ValueError):
            PredictionRequest(drugs=[], genotype=[StarAllele.parse("CYP2D6*1")])

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValueError):
            PredictionRequest(
                drugs=[2244],
                genotype=[StarAllele.parse("CYP2D6*1")],
                rogue_field="surprise",
            )


class TestTargetPrediction:
    def test_probability_in_range(self) -> None:
        tp = TargetPrediction(target="phenotype_category", label="poor", probability=0.87)
        assert tp.probability == 0.87
        assert tp.probabilities == {}

    @pytest.mark.parametrize("p", [-0.01, 1.01, 5.0])
    def test_probability_out_of_range_rejected(self, p: float) -> None:
        with pytest.raises(ValueError):
            TargetPrediction(target="x", label="y", probability=p)

    def test_full_distribution(self) -> None:
        tp = TargetPrediction(
            target="phenotype_category",
            label="normal",
            probability=0.6,
            probabilities={"poor": 0.1, "normal": 0.6, "rapid": 0.3},
        )
        assert sum(tp.probabilities.values()) == pytest.approx(1.0)


class TestPredictionResult:
    def test_construct(self) -> None:
        result = PredictionResult(
            model_name="TwoTowerGAT",
            model_version="0.7.0",
            predictions=[
                TargetPrediction(
                    target="phenotype_category", label="normal", probability=0.85
                ),
            ],
        )
        assert result.model_name == "TwoTowerGAT"
        assert len(result.predictions) == 1
        assert result.generated_at is not None  # default factory ran
