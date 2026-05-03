"""Liveness and readiness probes."""

from __future__ import annotations

from fastapi import APIRouter

from src.api.deps import RegistryDep, SettingsDep, default_model_name
from src.api.schemas import HealthResponse, ReadinessResponse


router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(settings: SettingsDep) -> HealthResponse:
    """Liveness probe — always returns 200 if the process is up."""
    return HealthResponse(
        status="ok",
        version=settings.version,
        project=settings.project_name,
    )


@router.get("/ready", response_model=ReadinessResponse)
def ready(settings: SettingsDep, registry: RegistryDep) -> ReadinessResponse:
    """Readiness probe — true once a model has been loaded.

    The default model is loaded lazily on first /v1/predict call. ``ready``
    will only return ``model_loaded=True`` after that has happened (or
    after a startup pre-load was performed).
    """
    name = default_model_name(settings)
    loaded = registry.is_loaded(name)
    return ReadinessResponse(
        status="ready" if loaded else "pending",
        model_loaded=loaded,
        model_name=name,
        detail=registry.last_error(name) if not loaded else None,
    )
