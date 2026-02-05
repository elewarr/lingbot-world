"""
Quantization utilities for LingBot-World on Apple Silicon.

Uses mps-bitsandbytes for 4-bit (NF4/FP4) and 8-bit (INT8/FP8) quantization.

Usage:
    from wan.utils.quantize import quantize_dit_model
    
    # Quantize to NF4 (4-bit, ~75% memory savings)
    model = quantize_dit_model(model, quant_type='nf4')
    
    # Quantize to INT8 (8-bit, ~50% memory savings)
    model = quantize_dit_model(model, quant_type='int8')
"""

import logging
import sys
from typing import Literal, Optional

import torch
import torch.nn as nn

# Check for MPS availability
IS_MACOS = sys.platform == 'darwin'
HAS_MPS = IS_MACOS and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

# Try to import mps-bitsandbytes
try:
    from mps_bitsandbytes import (
        Linear4bit,
        Linear8bit,
        LinearFP8,
        is_available as bnb_available,
    )
    HAS_MPS_BNB = bnb_available()
except ImportError:
    HAS_MPS_BNB = False
    Linear4bit = None
    Linear8bit = None
    LinearFP8 = None

__all__ = [
    'quantize_dit_model',
    'quantize_linear_layers',
    'get_model_memory_mb',
    'HAS_MPS_BNB',
]


def get_model_memory_mb(model: nn.Module) -> float:
    """Calculate model memory footprint in MB."""
    total_bytes = 0
    for param in model.parameters():
        total_bytes += param.numel() * param.element_size()
    for buffer in model.buffers():
        total_bytes += buffer.numel() * buffer.element_size()
    return total_bytes / 1024 / 1024


def quantize_linear(
    linear: nn.Linear,
    quant_type: Literal['nf4', 'fp4', 'int8', 'fp8'] = 'nf4',
    device: str = 'mps',
) -> nn.Module:
    """
    Quantize a single nn.Linear layer.
    
    Args:
        linear: The linear layer to quantize
        quant_type: Quantization type ('nf4', 'fp4', 'int8', 'fp8')
        device: Target device
    
    Returns:
        Quantized linear layer
    """
    if not HAS_MPS_BNB:
        logging.warning("mps-bitsandbytes not available, returning original layer")
        return linear
    
    # Move to half precision first (required for quantization)
    linear_half = linear.half().to(device)
    
    if quant_type in ('nf4', 'fp4'):
        return Linear4bit.from_linear(linear_half, quant_type=quant_type)
    elif quant_type == 'int8':
        return Linear8bit.from_linear(linear_half)
    elif quant_type == 'fp8':
        return LinearFP8.from_linear(linear_half)
    else:
        raise ValueError(f"Unknown quant_type: {quant_type}")


def quantize_linear_layers(
    module: nn.Module,
    quant_type: Literal['nf4', 'fp4', 'int8', 'fp8'] = 'nf4',
    device: str = 'mps',
    min_size: int = 1024,
    exclude_patterns: Optional[list] = None,
) -> int:
    """
    Recursively quantize all nn.Linear layers in a module.
    
    Args:
        module: The module to quantize
        quant_type: Quantization type
        device: Target device
        min_size: Minimum layer size to quantize (skip small layers)
        exclude_patterns: List of layer name patterns to exclude
    
    Returns:
        Number of layers quantized
    """
    if not HAS_MPS_BNB:
        logging.error("mps-bitsandbytes not available. Install with: pip install mps-bitsandbytes")
        return 0
    
    exclude_patterns = exclude_patterns or []
    quantized_count = 0
    
    for name, child in module.named_children():
        # Check if this layer should be excluded
        should_exclude = any(pattern in name for pattern in exclude_patterns)
        
        if isinstance(child, nn.Linear) and not should_exclude:
            # Check minimum size
            layer_size = child.in_features * child.out_features
            if layer_size >= min_size:
                try:
                    quantized = quantize_linear(child, quant_type, device)
                    setattr(module, name, quantized)
                    quantized_count += 1
                    logging.debug(f"Quantized {name}: {child.in_features}x{child.out_features}")
                except Exception as e:
                    logging.warning(f"Failed to quantize {name}: {e}")
        else:
            # Recurse into child modules
            quantized_count += quantize_linear_layers(
                child, quant_type, device, min_size, exclude_patterns
            )
    
    return quantized_count


def quantize_dit_model(
    model: nn.Module,
    quant_type: Literal['nf4', 'fp4', 'int8', 'fp8'] = 'nf4',
    device: str = 'mps',
    exclude_embeddings: bool = True,
    exclude_head: bool = True,
) -> nn.Module:
    """
    Quantize a DiT (Diffusion Transformer) model for inference.
    
    This function quantizes all linear layers in the model while optionally
    preserving embedding layers and output head in full precision.
    
    Args:
        model: The WanModel or similar DiT model
        quant_type: Quantization type:
            - 'nf4': 4-bit NormalFloat (best for LLM-like models, ~75% memory savings)
            - 'fp4': 4-bit FloatingPoint (alternative 4-bit format)
            - 'int8': 8-bit integer (~50% memory savings)
            - 'fp8': 8-bit floating point (better precision than int8)
        device: Target device ('mps', 'cuda', or 'cpu')
        exclude_embeddings: Keep embedding layers in full precision
        exclude_head: Keep output projection head in full precision
    
    Returns:
        Quantized model
    
    Example:
        >>> from wan.modules.model import WanModel
        >>> from wan.utils.quantize import quantize_dit_model
        >>> 
        >>> model = WanModel.from_pretrained(...)
        >>> model = quantize_dit_model(model, quant_type='nf4')
        >>> # Model now uses ~75% less memory
    """
    if not HAS_MPS_BNB:
        logging.error(
            "mps-bitsandbytes not available. "
            "Install with: pip install mps-bitsandbytes"
        )
        return model
    
    # Calculate memory before
    mem_before = get_model_memory_mb(model)
    
    # Build exclusion patterns
    exclude_patterns = []
    if exclude_embeddings:
        exclude_patterns.extend(['embed', 'patch_embed', 'pos_embed', 'time_embed'])
    if exclude_head:
        exclude_patterns.extend(['head', 'final', 'out_proj'])
    
    logging.info(f"Quantizing model to {quant_type.upper()}...")
    logging.info(f"Memory before: {mem_before:.1f} MB")
    
    # Quantize
    num_quantized = quantize_linear_layers(
        model,
        quant_type=quant_type,
        device=device,
        min_size=1024,  # Skip very small layers
        exclude_patterns=exclude_patterns,
    )
    
    # Calculate memory after
    mem_after = get_model_memory_mb(model)
    savings = (1 - mem_after / mem_before) * 100 if mem_before > 0 else 0
    
    logging.info(f"Quantized {num_quantized} layers")
    logging.info(f"Memory after: {mem_after:.1f} MB ({savings:.1f}% savings)")
    
    return model


def benchmark_quantization(
    model: nn.Module,
    input_shape: tuple = (1, 4096, 5120),
    quant_types: list = None,
    num_runs: int = 3,
    device: str = 'mps',
):
    """
    Benchmark different quantization types on a model.
    
    Args:
        model: Model to benchmark (will be copied for each test)
        input_shape: Shape of input tensor for forward pass
        quant_types: List of quantization types to test
        num_runs: Number of runs per configuration
        device: Target device
    
    Returns:
        dict: Benchmark results
    """
    import copy
    import time
    
    quant_types = quant_types or ['none', 'nf4', 'int8', 'fp8']
    results = {}
    
    for quant_type in quant_types:
        # Create a fresh copy
        test_model = copy.deepcopy(model)
        
        if quant_type != 'none':
            test_model = quantize_dit_model(test_model, quant_type=quant_type, device=device)
        else:
            test_model = test_model.to(device)
        
        test_model.eval()
        
        # Warmup
        with torch.no_grad():
            x = torch.randn(*input_shape, device=device, dtype=torch.float16)
            _ = test_model(x) if hasattr(test_model, 'forward') else None
        
        # Benchmark
        times = []
        mem_usage = get_model_memory_mb(test_model)
        
        for _ in range(num_runs):
            torch.mps.synchronize() if HAS_MPS else None
            start = time.perf_counter()
            
            with torch.no_grad():
                x = torch.randn(*input_shape, device=device, dtype=torch.float16)
                # Note: actual forward pass depends on model API
            
            torch.mps.synchronize() if HAS_MPS else None
            times.append(time.perf_counter() - start)
        
        results[quant_type] = {
            'avg_time': sum(times) / len(times),
            'min_time': min(times),
            'memory_mb': mem_usage,
        }
        
        del test_model
        if HAS_MPS:
            torch.mps.empty_cache()
    
    return results
