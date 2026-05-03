"""FastAPI application factory.

Run with::

    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

OpenAPI docs are served at ``/docs`` and ``/redoc``.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from src.api.deps import get_registry
from src.api.routers import health, library, models, predict
from src.config import configure_logging_level, get_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup / shutdown hooks.

    Configures logging from settings and ensures filesystem dirs exist. We
    deliberately *don't* pre-load the predictor here — model artifacts may be
    absent in dev/CI environments. Loading happens lazily on the first
    /v1/predict call (see PredictorRegistry).
    """
    settings = get_settings()
    configure_logging_level(settings)
    settings.paths.ensure_dirs()

    logger.info(
        "Pharmagen API starting (project=%s version=%s)",
        settings.project_name,
        settings.version,
    )
    app.state.settings = settings
    app.state.registry = get_registry()

    yield

    logger.info("Pharmagen API shutting down")


def create_app() -> FastAPI:
    """Build and return the FastAPI application.

    Use a factory rather than a module-level instance so tests can construct
    isolated apps and so the lifespan is explicit.
    """
    settings = get_settings()
    app = FastAPI(
        title=settings.project_name,
        version=settings.version,
        description=(
            "Pharmacogenetic prediction API. Maps a patient's drug list and "
            "star-allele genotype to per-target phenotypic predictions using "
            "the trained Two-Tower GATv2 model."
        ),
        lifespan=lifespan,
    )

    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(predict.router)
    app.include_router(library.router)

    return app


# Convenience: ``uvicorn src.api.main:app`` works without importing create_app.
app = create_app()
