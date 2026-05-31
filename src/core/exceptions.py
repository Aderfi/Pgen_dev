"""Domain exception hierarchy for Pharmagen.

All Pharmagen-specific exceptions inherit from ``PharmagenException`` so
callers can catch the whole family with a single ``except`` clause when
needed, while still being able to catch narrow subtypes for specific
handling.
"""

from collections.abc import Mapping
from typing import Any


class PharmagenException(Exception):
    """Base for all Pharmagen-specific errors."""

    def __init__(self, message: str, details: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            ctx = " | ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} | Context: {{{ctx}}}"
        return self.message


# ---------------------------------------------------------------------------
# Primary exceptions
# ---------------------------------------------------------------------------


class ConfigurationError(PharmagenException):
    """Invalid or missing configuration (hyperparameters, model settings)."""


class DataError(PharmagenException):
    """Invalid or incompatible data (missing columns, bad types, too few samples)."""


class ModelError(PharmagenException):
    """Model creation or loading failure (bad architecture, missing weights)."""


class PharmagenMemoryError(PharmagenException):
    """Memory constraint violated (OOM, cache overflow)."""


class OptimizationError(PharmagenException):
    """Optimization / hyperparameter search failure."""


class ResourceError(PharmagenException):
    """Insufficient system resources (disk, file handles, network)."""


class TrainingError(PharmagenException):
    """Training failure (NaN loss, gradient explosion, Optuna pruning)."""


class ValidationError(PharmagenException):
    """Input or configuration failed schema or range validation."""


# ---------------------------------------------------------------------------
# Derived exceptions
# ---------------------------------------------------------------------------


class ConvergenceError(TrainingError):
    """Training became numerically unstable (NaN / Inf loss or gradients)."""


class BioinformaticsError(DataError):
    """Genomic or VCF data is malformed or build-mismatched."""


class EncoderError(DataError):
    """Encoding / decoding failure (unknown category, missing fitted encoder)."""


class GraphError(DataError):
    """Graph data is invalid or corrupt (missing node features, bad edge indices)."""


class HardwareError(ResourceError):
    """GPU / CUDA / device error (OOM, device mismatch)."""
