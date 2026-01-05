# Pharmagen - Performance Monitoring Utilities
# Copyright (C) 2025 Adrim Hamed Outmani
#
# This module provides utilities for monitoring training performance
# and detecting bottlenecks during model training.

import logging
import time
from contextlib import contextmanager
from typing import Any, Dict

import torch

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    Monitors training performance metrics to identify bottlenecks.
    Optimized for RTX 4070 Ti SUPER + AMD Ryzen 7 7800X3D.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.timings: Dict[str, list] = {}
        self.gpu_memory: Dict[str, list] = {}

    @contextmanager
    def measure(self, section_name: str):
        """
        Context manager to measure execution time of code sections.
        
        Usage:
            monitor = PerformanceMonitor()
            with monitor.measure("data_loading"):
                data = load_data()
        """
        if not self.enabled:
            yield
            return

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.perf_counter()
        start_memory = (
            torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        )

        yield

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start_time
        memory_used = (
            (torch.cuda.memory_allocated() - start_memory) / 1024**2
            if torch.cuda.is_available()
            else 0
        )

        if section_name not in self.timings:
            self.timings[section_name] = []
            self.gpu_memory[section_name] = []

        self.timings[section_name].append(elapsed)
        self.gpu_memory[section_name].append(memory_used)

    def report(self) -> str:
        """Generate a performance report."""
        if not self.enabled or not self.timings:
            return "Performance monitoring disabled or no data collected."

        report_lines = [
            "=" * 70,
            "Performance Report",
            "=" * 70,
        ]

        for section_name, times in self.timings.items():
            avg_time = sum(times) / len(times)
            total_time = sum(times)
            calls = len(times)
            
            avg_memory = sum(self.gpu_memory[section_name]) / len(
                self.gpu_memory[section_name]
            )

            report_lines.append(f"\n{section_name}:")
            report_lines.append(f"  Calls:        {calls}")
            report_lines.append(f"  Avg Time:     {avg_time*1000:.2f} ms")
            report_lines.append(f"  Total Time:   {total_time:.2f} s")
            report_lines.append(f"  Avg GPU Mem:  {avg_memory:.2f} MB")

        report_lines.append("\n" + "=" * 70)
        return "\n".join(report_lines)

    def reset(self):
        """Reset all collected metrics."""
        self.timings.clear()
        self.gpu_memory.clear()


def log_gpu_info():
    """Log GPU information for performance debugging."""
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Running on CPU.")
        return

    device_count = torch.cuda.device_count()
    logger.info(f"Found {device_count} CUDA device(s)")

    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        logger.info(f"GPU {i}: {props.name}")
        logger.info(f"  Compute Capability: {props.major}.{props.minor}")
        logger.info(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")
        logger.info(f"  Multi-Processors: {props.multi_processor_count}")

    # Current memory usage
    current_device = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(current_device) / 1024**3
    reserved = torch.cuda.memory_reserved(current_device) / 1024**3
    
    logger.info(f"\nCurrent GPU Memory Usage (Device {current_device}):")
    logger.info(f"  Allocated: {allocated:.2f} GB")
    logger.info(f"  Reserved:  {reserved:.2f} GB")


def log_training_config():
    """Log current PyTorch configuration for performance optimization."""
    logger.info("PyTorch Configuration:")
    logger.info(f"  PyTorch version: {torch.__version__}")
    logger.info(f"  CUDA version: {torch.version.cuda}")
    logger.info(f"  cuDNN version: {torch.backends.cudnn.version()}")
    logger.info(f"  cuDNN enabled: {torch.backends.cudnn.enabled}")
    logger.info(f"  cuDNN benchmark: {torch.backends.cudnn.benchmark}")
    logger.info(f"  cuDNN deterministic: {torch.backends.cudnn.deterministic}")
    
    if hasattr(torch.backends.cuda, "matmul"):
        logger.info(f"  TF32 matmul: {torch.backends.cuda.matmul.allow_tf32}")
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        logger.info(f"  TF32 cuDNN: {torch.backends.cudnn.allow_tf32}")
    
    # Check for optimal settings
    warnings = []
    if torch.cuda.is_available():
        if not torch.backends.cudnn.benchmark:
            warnings.append("⚠ cuDNN benchmark is disabled. Enable for better performance.")
        if hasattr(torch.backends.cuda, "matmul") and not torch.backends.cuda.matmul.allow_tf32:
            warnings.append("⚠ TF32 matmul is disabled. Enable for faster training on Ampere+ GPUs.")
    
    if warnings:
        logger.warning("\nPerformance Warnings:")
        for warning in warnings:
            logger.warning(f"  {warning}")
    else:
        logger.info("\n✓ All performance optimizations enabled!")


def estimate_batch_size(model: torch.nn.Module, sample_input: tuple, max_memory_gb: float = 14.0):
    """
    Estimate optimal batch size based on available GPU memory.
    
    Args:
        model: The PyTorch model
        sample_input: Tuple of sample inputs (drug_data, haplo_data)
        max_memory_gb: Maximum GPU memory to use (leave headroom for other operations)
    
    Returns:
        Estimated optimal batch size
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, cannot estimate batch size")
        return 32  # Default fallback

    device = torch.cuda.current_device()
    
    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    
    model.eval()
    with torch.no_grad():
        # Run a single forward pass
        _ = model(*sample_input)
        
    # Get memory usage
    peak_memory = torch.cuda.max_memory_allocated(device) / 1024**3  # GB
    
    # Estimate batch size (accounting for gradients, optimizer states, etc.)
    # Rule of thumb: Training memory ≈ 3x forward pass memory
    # Breakdown: 1x forward + 1x backward (gradients) + 1x optimizer states (Adam/AdamW)
    # Note: This is a conservative estimate and may vary based on:
    #   - Model architecture (more complex = higher multiplier)
    #   - Optimizer choice (Adam uses 2x params for momentum, SGD uses less)
    #   - Mixed precision training (can reduce memory by ~2x)
    memory_per_sample = peak_memory * 3
    estimated_batch_size = int(max_memory_gb / memory_per_sample)
    
    # Ensure it's a reasonable value
    estimated_batch_size = max(1, min(estimated_batch_size, 256))
    
    logger.info(f"Memory per sample (estimated): {memory_per_sample*1024:.2f} MB")
    logger.info(f"Estimated optimal batch size: {estimated_batch_size}")
    
    return estimated_batch_size


def apply_performance_optimizations():
    """
    Apply all recommended PyTorch performance optimizations.
    Call this at the start of training.
    
    Optimized for NVIDIA RTX 4070 Ti SUPER (Ada Lovelace/Ampere architecture)
    and AMD Ryzen 7 7800X3D.
    """
    if torch.cuda.is_available():
        # Enable TF32 for faster matrix operations on Ampere and later
        if hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True
        
        # Use high precision matmul for better performance on modern GPUs
        # Available in PyTorch 1.12+
        try:
            torch.set_float32_matmul_precision('high')
        except AttributeError:
            pass  # Not available in older PyTorch versions (< 1.12)
        
        # Enable cuDNN benchmarking to find fastest convolution algorithms
        torch.backends.cudnn.benchmark = True
        
        logger.info("Applied CUDA performance optimizations")
    else:
        logger.warning("CUDA not available, skipping GPU optimizations")
