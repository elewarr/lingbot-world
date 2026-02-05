# wan/backends - Backend implementations for different frameworks
# MLX backend provides native Metal acceleration on Apple Silicon
# PyTorch backend provides the default implementation

from .base import (
    Backend,
    register_backend,
    get_backend,
    list_backends,
    get_available_backends,
    get_default_backend,
    mx_to_torch,
    torch_to_mx,
    mx_to_torch_list,
    torch_to_mx_list,
    convert_dit_cond_dict_to_mx,
    convert_dit_cond_dict_to_torch,
)

# Import backends to register them
from .pytorch import PyTorchBackend

# Conditionally import MLX backend (only on macOS)
import sys
if sys.platform == 'darwin':
    try:
        from .mlx.backend import MLXBackend
    except ImportError:
        # MLX not installed
        pass

__all__ = [
    # Base classes and registry
    'Backend',
    'register_backend',
    'get_backend',
    'list_backends',
    'get_available_backends',
    'get_default_backend',
    # Tensor transfer utilities
    'mx_to_torch',
    'torch_to_mx',
    'mx_to_torch_list',
    'torch_to_mx_list',
    'convert_dit_cond_dict_to_mx',
    'convert_dit_cond_dict_to_torch',
    # Backend implementations
    'PyTorchBackend',
]

# Add MLX to exports if available
if sys.platform == 'darwin':
    try:
        from .mlx.backend import MLXBackend
        __all__.append('MLXBackend')
    except ImportError:
        pass
