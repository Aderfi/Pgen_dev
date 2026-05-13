"""Cross-cutting primitives: exception hierarchy, logging setup, validators.

These modules have no dependencies on the rest of ``src/`` (except
``src.config``) and are imported broadly. Keeping them under a single
``core`` namespace mirrors the structure of ``src.config`` / ``src.domain``.
"""

from src.core.exceptions import (
    BioinformaticsError,
    ConfigurationError,
    ConvergenceError,
    DataError,
    EncoderError,
    GraphError,
    HardwareError,
    ModelError,
    OptimizationError,
    PharmagenException,
    PharmagenMemoryError,
    ResourceError,
    TrainingError,
    ValidationError,
)
from src.core.log import setup_logging
from src.core.validation import ConfigValidator, DataValidator

__all__ = [
    "BioinformaticsError",
    "ConfigValidator",
    "ConfigurationError",
    "ConvergenceError",
    "DataError",
    "DataValidator",
    "EncoderError",
    "GraphError",
    "HardwareError",
    "ModelError",
    "OptimizationError",
    "PharmagenException",
    "PharmagenMemoryError",
    "ResourceError",
    "TrainingError",
    "ValidationError",
    "setup_logging",
]
