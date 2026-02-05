"""Backend abstraction layer for hybrid PyTorch/MLX pipeline.

This module provides the abstract base class for backends and tensor transfer
utilities for bridging between PyTorch and MLX.

The hybrid architecture allows:
- VAE/T5 encoders to remain on PyTorch (mature, well-tested)
- DiT model to run on either PyTorch or MLX
- Efficient tensor conversion at model boundaries
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


# Lazy imports to avoid dependency issues
def _import_torch():
    import torch
    return torch


def _import_mlx():
    import mlx.core as mx
    return mx


class Backend(ABC):
    """Abstract base class for inference backends.
    
    Backends wrap framework-specific model implementations and provide
    a unified interface for the WanI2V pipeline.
    
    Implementations must handle:
    - Model loading from checkpoints
    - Forward pass execution
    - Tensor conversion to/from numpy
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier string."""
        pass
    
    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is usable on the current system."""
        pass
    
    @abstractmethod
    def load_model(
        self,
        checkpoint_path: str,
        model_config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        dtype: Optional[Any] = None,
    ) -> Any:
        """Load a model from checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint directory or file
            model_config: Optional model configuration overrides
            device: Target device ('cuda', 'mps', 'cpu', etc.)
            dtype: Model dtype (torch.float16, torch.bfloat16, etc.)
            
        Returns:
            Loaded model instance
        """
        pass
    
    @abstractmethod
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
        """Execute model forward pass.
        
        Args:
            model: Model instance (returned by load_model)
            x: Input tensor/array (list of video latents)
            t: Timestep tensor/array
            context: Text conditioning tensor/array (list of embeddings)
            seq_len: Maximum sequence length
            y: Optional conditional input for i2v mode
            dit_cond_dict: Optional camera conditioning dict
            
        Returns:
            Model output (denoised prediction)
        """
        pass
    
    @abstractmethod
    def to_numpy(self, tensor: Any) -> np.ndarray:
        """Convert backend tensor to numpy array.
        
        Args:
            tensor: Framework-specific tensor
            
        Returns:
            NumPy array copy of the data
        """
        pass
    
    @abstractmethod
    def from_numpy(
        self,
        array: np.ndarray,
        device: Optional[str] = None,
        dtype: Optional[Any] = None,
    ) -> Any:
        """Convert numpy array to backend tensor.
        
        Args:
            array: NumPy array
            device: Target device (backend-specific)
            dtype: Target dtype (backend-specific)
            
        Returns:
            Framework-specific tensor
        """
        pass
    
    def move_to_device(self, model: Any, device: str) -> Any:
        """Move model to specified device.
        
        Args:
            model: Model instance
            device: Target device string
            
        Returns:
            Model on target device
        """
        # Default implementation - subclasses may override
        return model
    
    def set_eval_mode(self, model: Any) -> Any:
        """Set model to evaluation mode.
        
        Args:
            model: Model instance
            
        Returns:
            Model in eval mode
        """
        # Default implementation - subclasses may override
        return model


# =============================================================================
# Tensor Transfer Utilities
# =============================================================================

def mx_to_torch(arr: Any, device: Optional[Any] = None) -> Any:
    """Convert MLX array to PyTorch tensor.
    
    This function evaluates the MLX array, converts to numpy,
    then creates a PyTorch tensor on the specified device.
    
    Args:
        arr: MLX array (mx.array)
        device: PyTorch device (torch.device or string)
        
    Returns:
        PyTorch tensor with the same data
        
    Example:
        >>> import mlx.core as mx
        >>> import torch
        >>> arr = mx.array([1.0, 2.0, 3.0])
        >>> tensor = mx_to_torch(arr, torch.device('cpu'))
    """
    mx = _import_mlx()
    torch = _import_torch()
    
    # Ensure computation is complete
    mx.eval(arr)
    
    # Convert via numpy (zero-copy when possible)
    np_arr = np.array(arr)
    tensor = torch.from_numpy(np_arr)
    
    if device is not None:
        tensor = tensor.to(device)
    
    return tensor


def torch_to_mx(tensor: Any) -> Any:
    """Convert PyTorch tensor to MLX array.
    
    This function moves the tensor to CPU if needed, converts to numpy,
    then creates an MLX array.
    
    Args:
        tensor: PyTorch tensor
        
    Returns:
        MLX array with the same data
        
    Example:
        >>> import torch
        >>> import mlx.core as mx
        >>> tensor = torch.tensor([1.0, 2.0, 3.0])
        >>> arr = torch_to_mx(tensor)
    """
    mx = _import_mlx()
    
    # Move to CPU and detach from computation graph
    np_arr = tensor.detach().cpu().numpy()
    
    # Create MLX array
    return mx.array(np_arr)


def mx_to_torch_list(
    arrs: List[Any],
    device: Optional[Any] = None,
) -> List[Any]:
    """Convert list of MLX arrays to PyTorch tensors.
    
    Args:
        arrs: List of MLX arrays
        device: Target PyTorch device
        
    Returns:
        List of PyTorch tensors
    """
    return [mx_to_torch(arr, device) for arr in arrs]


def torch_to_mx_list(tensors: List[Any]) -> List[Any]:
    """Convert list of PyTorch tensors to MLX arrays.
    
    Args:
        tensors: List of PyTorch tensors
        
    Returns:
        List of MLX arrays
    """
    return [torch_to_mx(t) for t in tensors]


def convert_dit_cond_dict_to_mx(
    dit_cond_dict: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Convert camera conditioning dict from PyTorch to MLX.
    
    Args:
        dit_cond_dict: Dictionary with PyTorch tensors or None
        
    Returns:
        Dictionary with MLX arrays or None
    """
    if dit_cond_dict is None:
        return None
    
    result = {}
    for key, value in dit_cond_dict.items():
        if key == "c2ws_plucker_emb" and value is not None:
            # value is a list of tensors
            result[key] = torch_to_mx_list(value)
        else:
            result[key] = value
    
    return result


def convert_dit_cond_dict_to_torch(
    dit_cond_dict: Optional[Dict[str, Any]],
    device: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """Convert camera conditioning dict from MLX to PyTorch.
    
    Args:
        dit_cond_dict: Dictionary with MLX arrays or None
        device: Target PyTorch device
        
    Returns:
        Dictionary with PyTorch tensors or None
    """
    if dit_cond_dict is None:
        return None
    
    result = {}
    for key, value in dit_cond_dict.items():
        if key == "c2ws_plucker_emb" and value is not None:
            # value is a list of arrays
            result[key] = mx_to_torch_list(value, device)
        else:
            result[key] = value
    
    return result


# =============================================================================
# Backend Registry
# =============================================================================

_BACKEND_REGISTRY: Dict[str, type] = {}


def register_backend(name: str):
    """Decorator to register a backend class.
    
    Args:
        name: Backend identifier string
        
    Example:
        @register_backend('pytorch')
        class PyTorchBackend(Backend):
            ...
    """
    def decorator(cls):
        _BACKEND_REGISTRY[name] = cls
        return cls
    return decorator


def get_backend(name: str) -> Backend:
    """Get a backend instance by name.
    
    Args:
        name: Backend identifier ('pytorch', 'mlx')
        
    Returns:
        Backend instance
        
    Raises:
        ValueError: If backend not found or not available
    """
    if name not in _BACKEND_REGISTRY:
        available = list(_BACKEND_REGISTRY.keys())
        raise ValueError(f"Unknown backend '{name}'. Available: {available}")
    
    backend_cls = _BACKEND_REGISTRY[name]
    backend = backend_cls()
    
    if not backend.is_available:
        raise ValueError(f"Backend '{name}' is not available on this system")
    
    return backend


def list_backends() -> List[str]:
    """List all registered backends.
    
    Returns:
        List of backend names
    """
    return list(_BACKEND_REGISTRY.keys())


def get_available_backends() -> List[str]:
    """List backends available on this system.
    
    Returns:
        List of available backend names
    """
    available = []
    for name, cls in _BACKEND_REGISTRY.items():
        try:
            backend = cls()
            if backend.is_available:
                available.append(name)
        except Exception:
            pass
    return available


def get_default_backend() -> str:
    """Get the default backend for the current system.
    
    Returns:
        Backend name ('pytorch' preferred, 'mlx' on Apple Silicon)
    """
    available = get_available_backends()
    
    # Prefer PyTorch for backward compatibility
    if 'pytorch' in available:
        return 'pytorch'
    elif 'mlx' in available:
        return 'mlx'
    else:
        raise RuntimeError("No backends available")
