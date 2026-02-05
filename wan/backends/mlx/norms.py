"""MLX implementations of normalization layers for WanModel.

This module provides MLX-native implementations of normalization layers
to enable Metal-accelerated inference on Apple Silicon.
"""

import mlx.core as mx
import mlx.nn as nn

__all__ = ['WanRMSNormMLX']


class WanRMSNormMLX(nn.Module):
    """Root Mean Square Layer Normalization for MLX.

    Implements the RMSNorm formula: x * rsqrt(mean(x^2) + eps) * weight

    This is a direct port of WanRMSNorm from wan/modules/model.py to MLX.
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

    def _norm(self, x: mx.array) -> mx.array:
        """Apply RMS normalization without the learnable weight.

        Formula: x * rsqrt(mean(x^2) + eps)
        """
        # Compute mean of squared values along the last dimension
        mean_sq = mx.mean(mx.square(x), axis=-1, keepdims=True)
        # Apply rsqrt normalization
        return x * mx.rsqrt(mean_sq + self.eps)

    def __call__(self, x: mx.array) -> mx.array:
        """Apply RMSNorm with learnable weight.

        Args:
            x: Input tensor of shape [..., dim]

        Returns:
            Normalized tensor of shape [..., dim]
        """
        # Convert to float32 for numerical stability, normalize, then scale
        x_float = x.astype(mx.float32)
        normed = self._norm(x_float)
        # Apply learnable weight and cast back to input dtype
        return (normed * self.weight).astype(x.dtype)
