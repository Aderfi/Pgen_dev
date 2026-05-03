"""Inference endpoints.

Single-pair predictions go through ``PGenPredictor.predict_single``; batches
loop in-process. A future enhancement is true vectorized batching via
``predict_file`` over an in-memory DataFrame.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, status

from src.api.deps import RegistryDep, SettingsDep, default_model_name
from src.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    SinglePredictRequest,
)
from src.domain.prediction import PredictionResult, TargetPrediction

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/predict", tags=["predict"])


def _request_to_features(req: SinglePredictRequest) -> dict[str, Any]:
    """Translate the typed API request into the dict shape the predictor wants.

    The TwoTowerGAT model expects features named ``drugs_cid`` and ``genotype``;
    that mapping is encoded here so the API contract stays clean even if model
    feature names drift.
    """
    return {"drugs_cid": str(req.drug_cid), "genotype": req.allele.label}


def _wrap_prediction(
    raw: dict[str, Any] | None,
    *,
    model_name: str,
    settings,
    request_id: str,
) -> PredictionResult:
    if raw is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="prediction returned no result (see server logs)",
        )

    targets: list[TargetPrediction] = []
    for target_name, label in raw.items():
        if isinstance(label, list):
            label_str = ", ".join(str(x) for x in label) if label else ""
        else:
            label_str = str(label)
        # The current predictor doesn't surface probabilities — Phase 4 will.
        targets.append(
            TargetPrediction(
                target=target_name,
                label=label_str,
                probability=1.0,  # placeholder until predictor exposes calibrated probs
            )
        )

    return PredictionResult(
        request_id=request_id,
        model_name=model_name,
        model_version=settings.version,
        predictions=targets,
    )


@router.post("", response_model=PredictionResult)
def predict(
    req: SinglePredictRequest,
    settings: SettingsDep,
    registry: RegistryDep,
) -> PredictionResult:
    name = default_model_name(settings)
    predictor = registry.get(name)

    request_id = uuid.uuid4().hex
    features = _request_to_features(req)
    raw = predictor.predict_single(features)
    return _wrap_prediction(raw, model_name=name, settings=settings, request_id=request_id)


@router.post("/batch", response_model=BatchPredictResponse)
def predict_batch(
    req: BatchPredictRequest,
    settings: SettingsDep,
    registry: RegistryDep,
) -> BatchPredictResponse:
    name = default_model_name(settings)
    predictor = registry.get(name)

    results: list[PredictionResult] = []
    for pair in req.pairs:
        request_id = uuid.uuid4().hex
        raw = predictor.predict_single(_request_to_features(pair))
        results.append(
            _wrap_prediction(
                raw, model_name=name, settings=settings, request_id=request_id
            )
        )

    return BatchPredictResponse(results=results, count=len(results))
