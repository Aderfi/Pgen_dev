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

"""Memory management utilities for Pharmagen.

This module provides tools for monitoring and managing memory usage
to prevent Out-Of-Memory (OOM) errors during training and Optuna optimization.
"""

import gc
import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """Monitor and manage memory usage for CPU and GPU.

    Provides utilities to track memory usage, clear caches,
    and prevent OOM errors during training.
    """

    @staticmethod
    def get_cpu_memory_mb() -> float:
        """Get current CPU memory usage in MB.

        Returns:
            Current process memory usage in megabytes.
        """
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            logger.warning("psutil not available, cannot monitor CPU memory")
            return 0.0

    @staticmethod
    def get_gpu_memory_mb(device: Optional[torch.device] = None) -> tuple[float, float]:
        """Get GPU memory usage in MB.

        Args:
            device: Target device (defaults to cuda:0 if available).

        Returns:
            Tuple of (allocated_mb, reserved_mb).
        """
        if not torch.cuda.is_available():
            return 0.0, 0.0

        if device is None:
            device = torch.device("cuda:0")

        allocated = torch.cuda.memory_allocated(device) / 1024 / 1024
        reserved = torch.cuda.memory_reserved(device) / 1024 / 1024
        return allocated, reserved

    @staticmethod
    def clear_memory(device: torch.device | None = None, aggressive: bool = False):
        """Clear cached memory and run garbage collection.

        Args:
            device: Target device for CUDA cache clearing.
            aggressive: If True, runs multiple GC passes and empties CUDA cache.
        """
        # CPU cleanup
        if aggressive:
            # Multiple GC passes can help with circular references
            for _ in range(3):
                gc.collect()
        else:
            gc.collect()

        # GPU cleanup
        if torch.cuda.is_available():
            if device is None:
                torch.cuda.empty_cache()
            else:
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()

        logger.debug("Memory cleared (aggressive=%s)", aggressive)

    @staticmethod
    def log_memory_stats(prefix: str = "", device: Optional[torch.device] = None):
        """Log current memory statistics.

        Args:
            prefix: Prefix for log message.
            device: Target device for GPU stats.
        """
        cpu_mem = MemoryMonitor.get_cpu_memory_mb()
        msg = f"{prefix}CPU: {cpu_mem:.1f}MB"

        if torch.cuda.is_available():
            gpu_alloc, gpu_reserved = MemoryMonitor.get_gpu_memory_mb(device)
            msg += f" | GPU Allocated: {gpu_alloc:.1f}MB, Reserved: {gpu_reserved:.1f}MB"

        logger.info(msg)

    @staticmethod
    def check_memory_available(
        required_mb: float,
        device: Optional[torch.device] = None,
        raise_error: bool = False
    ) -> bool:
        """Check if sufficient memory is available.

        Args:
            required_mb: Required memory in megabytes.
            device: Target device to check.
            raise_error: If True, raises RuntimeError when insufficient memory.

        Returns:
            True if sufficient memory available, False otherwise.

        Raises:
            RuntimeError: If raise_error=True and insufficient memory.
        """
        if device is not None and device.type == "cuda":
            if not torch.cuda.is_available():
                return True  # Skip check if CUDA not available

            total_mem = torch.cuda.get_device_properties(device).total_memory / 1024 / 1024
            allocated, reserved = MemoryMonitor.get_gpu_memory_mb(device)
            available = total_mem - reserved

            if available < required_mb:
                msg = (
                    f"Insufficient GPU memory: {available:.1f}MB available, "
                    f"{required_mb:.1f}MB required"
                )
                if raise_error:
                    raise RuntimeError(msg)
                logger.warning(msg)
                return False
        else:
            # For CPU, check system memory
            try:
                import psutil
                available_mb = psutil.virtual_memory().available / 1024 / 1024
                if available_mb < required_mb:
                    msg = (
                        f"Insufficient CPU memory: {available_mb:.1f}MB available, "
                        f"{required_mb:.1f}MB required"
                    )
                    if raise_error:
                        raise RuntimeError(msg)
                    logger.warning(msg)
                    return False
            except ImportError:
                logger.warning("psutil not available, skipping CPU memory check")

        return True


def estimate_model_memory_mb(
    num_parameters: int,
    dtype: torch.dtype = torch.float32,
    optimizer_factor: float = 2.0,
    gradient_factor: float = 1.0
) -> float:
    """Estimate memory required for a model.

    Args:
        num_parameters: Number of model parameters.
        dtype: Data type of parameters.
        optimizer_factor: Memory multiplier for optimizer state (2.0 for Adam).
        gradient_factor: Memory multiplier for gradients (1.0 typically).

    Returns:
        Estimated memory in megabytes.
    """
    bytes_per_param = torch.tensor([], dtype=dtype).element_size()
    # Model weights + gradients + optimizer state
    total_bytes = num_parameters * bytes_per_param * (1 + gradient_factor + optimizer_factor)
    return total_bytes / 1024 / 1024


def estimate_batch_memory_mb(
    batch_size: int,
    avg_nodes_per_graph: int,
    node_features: int,
    num_graphs: int = 2  # Drug + geno
) -> float:
    """Estimate memory required for a batch of graphs.

    Args:
        batch_size: Number of samples in batch.
        avg_nodes_per_graph: Average nodes per graph.
        node_features: Number of node features.
        num_graphs: Number of graphs per sample (default 2 for two-tower).

    Returns:
        Estimated memory in megabytes.
    """
    # Approximate: node features + edge index + edge attributes
    # Using float32 (4 bytes)
    total_nodes = batch_size * avg_nodes_per_graph * num_graphs
    node_memory = total_nodes * node_features * 4

    # Edge index (2 x num_edges, long = 8 bytes)
    # Assume avg 3 edges per node
    edge_memory = total_nodes * 3 * 2 * 8

    total_bytes = node_memory + edge_memory
    return total_bytes / 1024 / 1024
