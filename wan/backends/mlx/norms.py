"""MLX implementations of normalization layers for WanModel.

This module provides MLX-native implementations of normalization layers
to enable Metal-accelerated inference on Apple Silicon.
"""

import mlx.core as mx
import mlx.nn as nn

__all__ = ["WanRMSNormMLX", "WanLayerNormMLX"]


class WanRMSNormMLX(nn.Module):
    """Root Mean Square Layer Normalization for MLX.

    Implements the RMSNorm formula: x * rsqrt(mean(x^2) + eps) * weight

    This is a direct port of WanRMSNorm from wan/modules/model.py to MLX.
    Uses mx.fast.rms_norm for optimized Metal kernel performance.
    Used in self-attention for query/key normalization.

    Args:
        dim: Dimension of the input features (last axis size)
        eps: Small constant for numerical stability (default: 1e-5)
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        # Learnable scale parameter, initialized to ones
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        """Apply RMSNorm with learnable weight using fast fused kernel.

        Args:
            x: Input tensor of shape [..., dim]

        Returns:
            Normalized tensor of shape [..., dim]
        """
        # Use fast fused kernel for better performance
        # mx.fast.rms_norm computes in float32, cast back to preserve input dtype
        out = mx.fast.rms_norm(x, self.weight, self.eps)
        return out.astype(x.dtype) if out.dtype != x.dtype else out


class WanLayerNormMLX(nn.Module):
    """Layer Normalization for MLX.

    A direct port of WanLayerNorm from wan/modules/model.py to MLX.
    Uses mx.fast.layer_norm for optimized Metal kernel performance.
    Used in attention blocks for pre-normalization.

    Key differences from PyTorch nn.LayerNorm:
    - Computes in float32 for numerical stability
    - Casts result back to input dtype
    - Default eps=1e-6 (not 1e-5)
    - Default elementwise_affine=False (no learnable weight/bias)

    Args:
        dim: Dimension of the input features (last axis size)
        eps: Small constant for numerical stability (default: 1e-6)
        elementwise_affine: If True, adds learnable weight and bias (default: False)
    """

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = False):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        # Initialize weight and bias if using affine transform
        if elementwise_affine:
            self.weight = mx.ones((dim,))
            self.bias = mx.zeros((dim,))
        else:
            self.weight = None
            self.bias = None

    def __call__(self, x: mx.array) -> mx.array:
        """Apply LayerNorm using fast fused kernel.

        Args:
            x: Input tensor of shape [..., dim]

        Returns:
            Normalized tensor of shape [..., dim]
        """
        # Use fast fused kernel for better performance
        # mx.fast.layer_norm handles float32 conversion internally
        return mx.fast.layer_norm(x, self.weight, self.bias, self.eps)
