# Pharmagen - Pharmacogenetic Prediction and Therapeutic Efficacy
# Copyright (C) 2025 Adrim Hamed Outmani
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Custom exceptions for Pharmagen.

This module defines a hierarchy of custom exceptions to provide
clear, actionable error messages and improve error handling.

Follows best practices:
- Explicit is better than implicit
- Errors should never pass silently
- In the face of ambiguity, refuse the temptation to guess
"""

from collections.abc import Mapping
from typing import Any


#### Base Exception ####
class PharmagenException(Exception):
    """Base exception for all Pharmagen-specific errors.

    All custom exceptions should inherit from this base class
    to allow catching all Pharmagen errors with a single except clause.
    """
    def __init__(self, message: str, details: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self):
        if self.details:
            details_str = "|".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} | Context: {{{details_str}}}"
        return self.message

# DERIVED EXCEPTIONS #
# SECONDARY EXCEPTION CLASSES #

class ConfigurationError(PharmagenException):
    """Raised when configuration is invalid or missing.

    Examples:
        - Missing required configuration keys
        - Invalid hyperparameter values
        - Incompatible model settings
    """
    pass

class DataError(PharmagenException):
    """Raised when data is invalid or incompatible.

    Examples:
        - Missing required columns
        - Invalid data types
        - Insufficient samples for training
    """
    pass

class ModelError(PharmagenException):
    """Raised when model creation or loading fails.

    Examples:
        - Invalid model architecture
        - Incompatible model weights
        - Missing model files
    """
    pass

class PharmagenMemoryError(PharmagenException):
    """Raised when memory constraints are violated.

    Examples:
        - Insufficient memory for batch size
        - OOM during training
        - Cache size exceeds limits
    """
    pass


# Backwards-compatible alias. New code should use PharmagenMemoryError to
# avoid shadowing the built-in.
MemoryError = PharmagenMemoryError

class OptimizationError(PharmagenException):
    """Raised when optimization/training fails.

    Examples:
        - Loss becomes NaN
        - Gradient explosion
        - Optuna trial failures
    """
    pass

class ResourceError(PharmagenException):
    """Raised when system resources are insufficient.

    Examples:
        - Disk space exhausted
        - File handle limits exceeded
        - Network connectivity issues
    """
    pass

class TrainingError(PharmagenException):
    """Raised when optimization/training fails.

    Examples:
        - Loss becomes NaN
        - Gradient explosion
        - Optuna trial failures
    """
    pass

class ValidationError(PharmagenException):
    """Raised when validation of inputs or configurations fails.

    Examples:
        - Missing required fields
        - Invalid parameter values
        - Data schema mismatches
    """
    pass

# TERTIARY EXCEPTION CLASSES #

class ConvergenceError(TrainingError):
    """Raised when training becomes unstable.

    Examples:
        - Loss is NaN or Inf
        - Gradient explosion detected
    """
    pass

class BioinformaticsError(DataError):
    """Raised when bio-formats or genomic data are invalid.

    Examples:
        - VCF file parsing failure
        - Mismatch between FASTA and GFF
        - RSID not found in database
    """
    pass

class EncoderError(DataError):
    """Raised when encoding/decoding fails.

    Examples:
        - Unknown category in test data
        - Missing fitted encoder
        - Incompatible encoder classes
    """
    pass

class GraphError(DataError):
    """Raised when graph data is invalid or corrupt.

    Examples:
        - Missing node features
        - Invalid edge indices
        - Corrupt graph files
    """
    pass

class HardwareError(ResourceError):
    """Raised specifically for GPU/CUDA/MPS issues.

    Examples:
        - CUDA OOM (Out Of Memory)
        - Device mismatch (Tensor on CPU vs Model on GPU)
    """
    pass
