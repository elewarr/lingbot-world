"""MLX backend implementation.

This module wraps WanModelMLX for use with the backend abstraction.
It provides Metal-accelerated inference on Apple Silicon.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

# Check if MLX is available (Apple Silicon only)
_MLX_AVAILABLE = False
try:
    if sys.platform == 'darwin':
        import mlx.core as mx
        import mlx.nn as nn
        _MLX_AVAILABLE = True
except ImportError:
    pass

from ..base import Backend, register_backend, torch_to_mx, mx_to_torch


@register_backend('mlx')
class MLXBackend(Backend):
    """MLX backend wrapping WanModelMLX.
    
    This backend uses the MLX implementation for Metal-accelerated
    inference on Apple Silicon. It supports automatic weight conversion
    from PyTorch checkpoints and optional quantization.
    
    Features:
    - Native Metal acceleration
    - Automatic PyTorch weight conversion with caching
    - 4-bit and 8-bit quantization support
    - Memory-efficient inference
    """
    
    def __init__(self, quantization_bits: Optional[int] = None):
        """Initialize MLX backend.
        
        Args:
            quantization_bits: Optional quantization (4 or 8 bits)
        """
        self._quantization_bits = quantization_bits
    
    @property
    def name(self) -> str:
        return 'mlx'
    
    @property
    def is_available(self) -> bool:
        """Check if MLX is available (Apple Silicon only)."""
        return _MLX_AVAILABLE
    
    def load_model(
        self,
        checkpoint_path: str,
        model_config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,  # Ignored for MLX (always uses Metal)
        dtype: Optional[Any] = None,
    ) -> Any:
        """Load WanModelMLX from checkpoint.
        
        Supports loading from:
        - PyTorch checkpoints (auto-converted)
        - Cached MLX weights
        - Quantized models
        
        Args:
            checkpoint_path: Path to model checkpoint directory
            model_config: Optional config overrides including:
                - subfolder: Subfolder within checkpoint_path
                - quantize: Quantization bits (4 or 8)
                - use_cache: Whether to use weight cache
            device: Ignored (MLX always uses Metal)
            dtype: Model dtype (mx.float16 or mx.float32)
            
        Returns:
            WanModelMLX instance with loaded weights
        """
        if not _MLX_AVAILABLE:
            raise RuntimeError("MLX is not available. Requires macOS on Apple Silicon.")
        
        from ..mlx.model import WanModelMLX
        from ..mlx.quantize import quantize_model, load_quantized_model
        
        config = model_config or {}
        subfolder = config.get('subfolder', None)
        quantize = config.get('quantize', self._quantization_bits)
        use_cache = config.get('use_cache', True)
        
        checkpoint_path = Path(checkpoint_path)
        if subfolder:
            checkpoint_path = checkpoint_path / subfolder
        
        logging.info(f"MLXBackend: Loading model from {checkpoint_path}")
        if quantize:
            logging.info(f"  quantization: {quantize}-bit")
        
        # Check for pre-quantized model
        quantized_path = checkpoint_path / f"mlx_quantized_{quantize}bit.safetensors"
        if quantize and quantized_path.exists():
            logging.info(f"  Loading pre-quantized weights from {quantized_path}")
            model = load_quantized_model(str(quantized_path))
        else:
            # Load from PyTorch checkpoint
            model = WanModelMLX.from_pretrained(
                str(checkpoint_path),
                use_cache=use_cache,
            )
            
            # Apply quantization if requested
            if quantize:
                logging.info(f"  Quantizing model to {quantize}-bit...")
                model = quantize_model(model, bits=quantize)
        
        # Evaluate to ensure weights are loaded
        mx.eval(model.parameters())
        
        return model
    
    def forward(
        self,
        model: Any,
        x: Any,
        t: Any,
        context: Any,
        seq_len: int,
        y: Optional[Any] = None,
        dit_cond_dict: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Execute WanModelMLX forward pass.
        
        Handles tensor conversion from PyTorch if needed.
        
        Args:
            model: WanModelMLX instance
            x: List of input tensors (PyTorch or MLX)
            t: Timestep tensor/array
            context: List of text embeddings
            seq_len: Maximum sequence length
            y: Optional conditional input
            dit_cond_dict: Camera conditioning dict
            
        Returns:
            List of denoised video arrays
        """
        if not _MLX_AVAILABLE:
            raise RuntimeError("MLX is not available")
        
        import torch
        from ..base import convert_dit_cond_dict_to_mx
        
        # Convert inputs if they're PyTorch tensors
        if isinstance(x, list) and len(x) > 0:
            if isinstance(x[0], torch.Tensor):
                x = [torch_to_mx(u) for u in x]
        
        if isinstance(t, torch.Tensor):
            t = torch_to_mx(t)
        
        if isinstance(context, list) and len(context) > 0:
            if isinstance(context[0], torch.Tensor):
                context = [torch_to_mx(c) for c in context]
        
        if y is not None and isinstance(y, list) and len(y) > 0:
            if isinstance(y[0], torch.Tensor):
                y = [torch_to_mx(u) for u in y]
        
        # Convert camera conditioning
        if dit_cond_dict is not None:
            dit_cond_dict = convert_dit_cond_dict_to_mx(dit_cond_dict)
        
        # Run forward pass
        output = model(
            x=x,
            t=t,
            context=context,
            seq_len=seq_len,
            y=y,
            dit_cond_dict=dit_cond_dict,
        )
        
        # Ensure computation is complete
        mx.eval(output)
        
        return output
    
    def to_numpy(self, tensor: Any) -> np.ndarray:
        """Convert MLX array to numpy.
        
        Args:
            tensor: MLX array
            
        Returns:
            NumPy array
        """
        if not _MLX_AVAILABLE:
            raise RuntimeError("MLX is not available")
        
        mx.eval(tensor)
        return np.array(tensor)
    
    def from_numpy(
        self,
        array: np.ndarray,
        device: Optional[str] = None,  # Ignored for MLX
        dtype: Optional[Any] = None,
    ) -> Any:
        """Convert numpy array to MLX array.
        
        Args:
            array: NumPy array
            device: Ignored (MLX always uses Metal)
            dtype: Target MLX dtype
            
        Returns:
            MLX array
        """
        if not _MLX_AVAILABLE:
            raise RuntimeError("MLX is not available")
        
        arr = mx.array(array)
        
        if dtype is not None:
            arr = arr.astype(dtype)
        
        return arr
    
    def set_eval_mode(self, model: Any) -> Any:
        """Set model to evaluation mode.
        
        MLX models don't have a training mode, so this is a no-op.
        
        Args:
            model: WanModelMLX instance
            
        Returns:
            Same model
        """
        # MLX models are always in eval mode
        return model
    
    def convert_output_to_torch(
        self,
        output: Any,
        device: Optional[Any] = None,
    ) -> Any:
        """Convert MLX output to PyTorch tensors.
        
        Useful for hybrid pipelines where VAE decode runs on PyTorch.
        
        Args:
            output: List of MLX arrays
            device: Target PyTorch device
            
        Returns:
            List of PyTorch tensors
        """
        if isinstance(output, list):
            return [mx_to_torch(o, device) for o in output]
        else:
            return mx_to_torch(output, device)
    
    def synchronize(self) -> None:
        """Synchronize Metal operations (wait for completion)."""
        if _MLX_AVAILABLE:
            mx.eval([])  # Barrier to ensure all operations complete


def get_mlx_memory_info() -> Dict[str, float]:
    """Get MLX memory usage information.
    
    Returns:
        Dictionary with memory stats in GB:
        - 'active': Currently allocated memory
        - 'peak': Peak memory usage
        - 'cache': Memory in cache
    """
    if not _MLX_AVAILABLE:
        return {'active': 0.0, 'peak': 0.0, 'cache': 0.0}
    
    # MLX doesn't have built-in memory tracking like PyTorch
    # Return placeholder values
    return {
        'active': 0.0,
        'peak': 0.0,
        'cache': 0.0,
    }
