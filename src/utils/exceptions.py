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


class PharmagenException(Exception):
    """Base exception for all Pharmagen-specific errors.

    All custom exceptions should inherit from this base class
    to allow catching all Pharmagen errors with a single except clause.
    """
    pass


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


class MemoryError(PharmagenException):
    """Raised when memory constraints are violated.

    Examples:
        - Insufficient memory for batch size
        - OOM during training
        - Cache size exceeds limits

    Note:
        This shadows Python's built-in MemoryError.
        Use fully qualified name if needed: builtins.MemoryError
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


class EncoderError(DataError):
    """Raised when encoding/decoding fails.
    
    Examples:
        - Unknown category in test data
        - Missing fitted encoder
        - Incompatible encoder classes
    """
    pass


class OptimizationError(PharmagenException):
    """Raised when optimization/training fails.

    Examples:
        - Loss becomes NaN
        - Gradient explosion
        - Optuna trial failures
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
