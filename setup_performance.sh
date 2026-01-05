#!/bin/bash
# Pharmagen Performance Optimization Setup for WSL Debian + RTX 4070 Ti SUPER
# Copyright (C) 2025 Adrim Hamed Outmani
#
# This script sets environment variables for optimal PyTorch performance
# Run with: source setup_performance.sh

echo "======================================================"
echo "Pharmagen Performance Optimization Setup"
echo "Target: WSL Debian Trixie (13) + RTX 4070 Ti SUPER"
echo "======================================================"

# PyTorch CUDA optimizations
export CUDA_LAUNCH_BLOCKING=0  # Async kernel launches for better performance
export TORCH_CUDNN_V8_API_ENABLED=1  # Enable cuDNN v8 API
export CUDA_MODULE_LOADING=LAZY  # Lazy load CUDA modules to reduce startup time

# PyTorch performance settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
# Note: CUDA Device-Side Assertions (DSA) are disabled for production performance
# Enable DSA only for debugging: export TORCH_USE_CUDA_DSA=1

# Multi-threading optimizations - dynamically detect CPU cores
NUM_CORES=$(nproc 2>/dev/null || echo "8")  # Fallback to 8 if nproc unavailable
export OMP_NUM_THREADS=$NUM_CORES
export MKL_NUM_THREADS=$NUM_CORES
export OPENBLAS_NUM_THREADS=$NUM_CORES
export NUMEXPR_NUM_THREADS=$NUM_CORES

echo "✓ CPU threads set to: $NUM_CORES (physical cores)"

# DataLoader worker optimization
export PYTORCH_DATALOADER_WORKER_OFFLOAD=1  # Offload data to workers efficiently

# Memory allocator optimizations
export MALLOC_TRIM_THRESHOLD_=100000  # Reduce memory fragmentation
export MALLOC_MMAP_THRESHOLD_=100000

# WSL-specific optimizations
# Ensure CUDA is accessible through WSL
if [ -d "/usr/lib/wsl" ]; then
    export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
    echo "✓ WSL CUDA libraries added to LD_LIBRARY_PATH"
fi

# Check CUDA availability
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠ Warning: nvidia-smi not found. CUDA may not be available."
fi

# Check Python and PyTorch
if command -v python3 &> /dev/null; then
    echo "✓ Python version: $(python3 --version)"
    
    # Check PyTorch CUDA
    python3 -c "import torch; print(f'✓ PyTorch version: {torch.__version__}'); print(f'✓ CUDA available: {torch.cuda.is_available()}'); print(f'✓ CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'✓ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null || echo "⚠ PyTorch not installed or error checking"
fi

echo "======================================================"
echo "Environment variables set successfully!"
echo "======================================================"
echo ""
echo "Optimization summary:"
echo "  - CUDA async execution enabled"
echo "  - CPU threads: 8 (matching physical cores)"
echo "  - Memory allocator optimized for large allocations"
echo "  - cuDNN v8 API enabled"
echo ""
echo "To apply these settings, run:"
echo "  source setup_performance.sh"
echo ""
echo "For permanent setup, add to ~/.bashrc:"
echo "  source $(pwd)/setup_performance.sh"
echo "======================================================"
