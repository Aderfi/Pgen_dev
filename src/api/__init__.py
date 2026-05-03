"""Pharmagen FastAPI service.

Public entrypoint:
    >>> from src.api.main import create_app
    >>> app = create_app()

Run with:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000
"""

from src.api.main import create_app


__all__ = ["create_app"]
