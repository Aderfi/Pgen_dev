"""FastAPI dependency-injection helpers.

The predictor is loaded lazily — the API can boot and serve /health without
trained model artifacts. The first /v1/predict request triggers a load and
caches it for subsequent requests; if loading fails, callers see 503.
"""

from __future__ import annotations

import logging
from threading import Lock
from typing import Annotated

from fastapi import Depends, HTTPException, status

from src.config import Settings, get_settings


logger = logging.getLogger(__name__)


SettingsDep = Annotated[Settings, Depends(get_settings)]


class PredictorRegistry:
    """Per-process cache of loaded predictors keyed by model name.

    The PGenPredictor takes seconds to instantiate (loads encoders + weights),
    so we never want to do it more than once per (process, model_name).
    """

    def __init__(self) -> None:
        self._predictors: dict[str, object] = {}
        self._lock = Lock()
        self._last_error: dict[str, str] = {}

    def get(self, model_name: str):
        """Return the predictor for ``model_name``, loading on first request.

        Raises HTTPException(503) if the model artifacts are missing — that's
        the expected failure mode when the API boots without a trained model.
        """
        if model_name in self._predictors:
            return self._predictors[model_name]

        with self._lock:
            if model_name in self._predictors:
                return self._predictors[model_name]

            try:
                # Local import: avoids pulling torch into /health response time.
                from src.modeling.engine.predictor import PGenPredictor

                predictor = PGenPredictor(model_name)
            except FileNotFoundError as e:
                self._last_error[model_name] = str(e)
                logger.warning("Model %s not loadable: %s", model_name, e)
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=(
                        f"Model {model_name!r} is not available — required "
                        f"artifact missing: {e}"
                    ),
                ) from e
            except Exception as e:
                self._last_error[model_name] = str(e)
                logger.exception("Failed to load model %s", model_name)
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Model {model_name!r} failed to load: {e}",
                ) from e

            self._predictors[model_name] = predictor
            return predictor

    def is_loaded(self, model_name: str) -> bool:
        return model_name in self._predictors

    def loaded_models(self) -> list[str]:
        return list(self._predictors)

    def last_error(self, model_name: str) -> str | None:
        return self._last_error.get(model_name)


# Single registry per process. FastAPI's lifespan attaches it to app.state, but
# we expose a module-level instance for code that doesn't have an app handle
# (e.g. unit tests).
_registry = PredictorRegistry()


def get_registry() -> PredictorRegistry:
    return _registry


RegistryDep = Annotated[PredictorRegistry, Depends(get_registry)]


def default_model_name(settings: Settings) -> str:
    """The model the API serves by default. For now, the first model in the
    catalog — multi-model serving is a future enhancement."""
    from src.config import get_available_models

    available = get_available_models()
    if not available:
        msg = "No models defined in models.toml"
        raise RuntimeError(msg)
    return available[0]
