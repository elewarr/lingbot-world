"""MLX implementation of Head (output projection) for WanModel.

This module provides an MLX-native implementation of the output projection head
to enable Metal-accelerated inference on Apple Silicon.

Key features:
- Layer normalization followed by linear projection
- 2-way modulation (scale and shift) from time embedding
- Output dimension = prod(patch_size) * out_dim

Reference: wan/modules/model.py (Head class)
"""

import math
from typing import Any, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .norms import WanLayerNormMLX

__all__ = ['HeadMLX']


class HeadMLX(nn.Module):  # type: ignore[misc, name-defined]
    """Output projection head with modulation for MLX.

    This is a direct port of Head from wan/modules/model.py to MLX.
    Projects transformer hidden states to output latent space with
    2-way modulation from time embeddings.

    Args:
        dim: Hidden dimension of the model (e.g., 2048)
        out_dim: Output channels (e.g., 16 for latent channels)
        patch_size: Tuple of (t, h, w) for temporal/spatial patch sizes
        eps: Epsilon for layer normalization
    """

    def __init__(
        self,
        dim: int,
        out_dim: int,
        patch_size: Union[Tuple[int, int, int], Tuple[int, ...]],
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # Calculate output dimension: prod(patch_size) * out_dim
        # For patch_size=(1, 2, 2) and out_dim=16: 1*2*2*16 = 64
        full_out_dim = math.prod(patch_size) * out_dim

        # Layer normalization (no learnable params by default)
        self.norm = WanLayerNormMLX(dim, eps=eps, elementwise_affine=False)

        # Output projection linear layer
        self.head: Any = nn.Linear(dim, full_out_dim)  # type: ignore[attr-defined]

        # Modulation parameter (1, 2, dim) - splits into 2 parts for scale/shift
        # Initialize with randn / sqrt(dim) to match PyTorch
        self.modulation = mx.random.normal((1, 2, dim)) / (dim ** 0.5)

    def __call__(self, x: mx.array, e: mx.array) -> mx.array:
        """Apply output projection with modulation.

        Args:
            x: Hidden states of shape [B, L, C] where C = dim
            e: Time embedding of shape [B, L, C] (note: no chunk dimension)

        Returns:
            Output tensor of shape [B, L, full_out_dim] where
            full_out_dim = prod(patch_size) * out_dim
        """
        # Modulation computation (following PyTorch exactly):
        # 1. e.unsqueeze(2) -> [B, L, 1, C]
        # 2. modulation.unsqueeze(0) -> [1, 1, 2, C]
        # 3. Add them: [B, L, 2, C]
        # 4. chunk(2, dim=2) -> 2 x [B, L, 1, C]
        # 5. squeeze(2) each -> [B, L, C]

        # Expand dims: e [B, L, C] -> [B, L, 1, C]
        e_expanded = mx.expand_dims(e, axis=2)

        # Expand modulation: [1, 2, C] -> [1, 1, 2, C]
        mod_expanded = mx.expand_dims(self.modulation, axis=0)

        # Add and split: [B, L, 2, C]
        e_mod = mod_expanded + e_expanded

        # Split along axis 2 into shift (e0) and scale (e1)
        # e0 = e_mod[:, :, 0:1, :] squeezed -> [B, L, C]
        # e1 = e_mod[:, :, 1:2, :] squeezed -> [B, L, C]
        e0 = mx.squeeze(e_mod[:, :, 0:1, :], axis=2)  # shift
        e1 = mx.squeeze(e_mod[:, :, 1:2, :], axis=2)  # scale

        # Apply modulation and projection:
        # x_modulated = norm(x) * (1 + scale) + shift
        # output = head(x_modulated)
        x_normed = self.norm(x)
        x_modulated = x_normed * (1.0 + e1) + e0

        output: mx.array = self.head(x_modulated)

        return output
