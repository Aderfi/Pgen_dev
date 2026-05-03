"""Model catalog endpoints — list available trained models and their config."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from src.api.schemas import ModelDetail, ModelSummary
from src.config import get_available_models, get_model_config


router = APIRouter(prefix="/v1/models", tags=["models"])


@router.get("", response_model=list[ModelSummary])
def list_models() -> list[ModelSummary]:
    """List the names defined in ``models.toml`` with their feature/target spec."""
    summaries: list[ModelSummary] = []
    for name in get_available_models():
        cfg = get_model_config(name)
        summaries.append(
            ModelSummary(name=cfg.name, features=cfg.features, targets=cfg.targets)
        )
    return summaries


@router.get("/{name}", response_model=ModelDetail)
def get_model(name: str) -> ModelDetail:
    try:
        cfg = get_model_config(name)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(e)
        ) from e

    return ModelDetail(
        name=cfg.name,
        features=cfg.features,
        targets=cfg.targets,
        cols=cfg.cols,
        stratify_col=cfg.stratify_col,
        fixed_params=cfg.fixed_params(),
        optuna_keys=list(cfg.optuna.keys()),
        extras=cfg.extras,
    )
