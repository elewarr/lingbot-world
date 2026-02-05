"""MLX implementation of WanSelfAttention for WanModel.

This module provides an MLX-native implementation of the self-attention layer
to enable Metal-accelerated inference on Apple Silicon.

Key differences from PyTorch implementation:
- Uses mlx.nn.Linear instead of torch.nn.Linear
- Uses mx.fast.scaled_dot_product_attention instead of flash_attention
- Uses rope_apply_mlx instead of rope_apply

Reference: wan/modules/model.py (WanSelfAttention class)
"""

from typing import Tuple, Optional, Any

import mlx.core as mx
import mlx.nn as nn

from .norms import WanRMSNormMLX
from .rope import rope_apply_mlx

__all__ = ['WanSelfAttentionMLX']


class WanSelfAttentionMLX(nn.Module):  # type: ignore[misc, name-defined]
    """Self-attention layer with QK normalization and RoPE for MLX.

    This is a direct port of WanSelfAttention from wan/modules/model.py to MLX.
    Supports multi-head attention with optional query/key normalization and
    rotary position embeddings (RoPE) for 3D spatial-temporal positions.

    Args:
        dim: Hidden dimension of the model (e.g., 2048)
        num_heads: Number of attention heads (e.g., 16)
        window_size: Window size for local attention. (-1, -1) means global attention.
        qk_norm: If True, apply RMSNorm to query and key projections
        eps: Epsilon for normalization layers
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int] = (-1, -1),
        qk_norm: bool = True,
        eps: float = 1e-6,
    ) -> None:
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # Compute scale for attention (1/sqrt(head_dim))
        self.scale = self.head_dim ** -0.5

        # Linear projections for Q, K, V, and output
        self.q: Any = nn.Linear(dim, dim)  # type: ignore[attr-defined]
        self.k: Any = nn.Linear(dim, dim)  # type: ignore[attr-defined]
        self.v: Any = nn.Linear(dim, dim)  # type: ignore[attr-defined]
        self.o: Any = nn.Linear(dim, dim)  # type: ignore[attr-defined]

        # Optional QK normalization
        self.norm_q: Optional[WanRMSNormMLX] = WanRMSNormMLX(dim, eps=eps) if qk_norm else None
        self.norm_k: Optional[WanRMSNormMLX] = WanRMSNormMLX(dim, eps=eps) if qk_norm else None

    def __call__(
        self,
        x: mx.array,
        seq_lens: mx.array,
        grid_sizes: mx.array,
        freqs_cos: Tuple[mx.array, mx.array, mx.array],
        freqs_sin: Tuple[mx.array, mx.array, mx.array],
    ) -> mx.array:
        """Forward pass of the self-attention layer.

        Args:
            x: Input tensor of shape [B, L, C] where:
                B = batch size
                L = sequence length (padded to max)
                C = hidden dimension (self.dim)
            seq_lens: Tensor of shape [B] containing the actual sequence length
                for each sample in the batch
            grid_sizes: Tensor of shape [B, 3] containing (F, H, W) grid dimensions
                for each sample, used for 3D RoPE
            freqs_cos: Tuple of 3 cosine frequency tensors for (frame, height, width)
            freqs_sin: Tuple of 3 sine frequency tensors for (frame, height, width)

        Returns:
            Output tensor of shape [B, L, C]
        """
        b, s, _ = x.shape
        n, d = self.num_heads, self.head_dim

        # Compute Q, K, V projections
        # Shape: [B, L, dim] -> [B, L, dim]
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        # Apply QK normalization if enabled
        if self.qk_norm and self.norm_q is not None and self.norm_k is not None:
            q = self.norm_q(q)
            k = self.norm_k(k)

        # Reshape to [B, L, num_heads, head_dim]
        q = q.reshape(b, s, n, d)
        k = k.reshape(b, s, n, d)
        v = v.reshape(b, s, n, d)

        # Apply rotary position embeddings
        q = rope_apply_mlx(q, grid_sizes, freqs_cos, freqs_sin)
        k = rope_apply_mlx(k, grid_sizes, freqs_cos, freqs_sin)

        # Transpose to [B, num_heads, L, head_dim] for attention
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))

        # Create attention mask for variable sequence lengths
        # For global attention (window_size == (-1, -1)), we just need to mask padding
        mask = self._create_attention_mask(seq_lens, s)

        # Compute scaled dot-product attention using MLX's optimized implementation
        # mx.fast.scaled_dot_product_attention handles the scaling internally
        attn_output = mx.fast.scaled_dot_product_attention(
            q, k, v,
            scale=self.scale,
            mask=mask,
        )

        # Transpose back to [B, L, num_heads, head_dim]
        attn_output = mx.transpose(attn_output, (0, 2, 1, 3))

        # Flatten heads: [B, L, num_heads, head_dim] -> [B, L, dim]
        attn_output = attn_output.reshape(b, s, -1)

        # Output projection
        output = self.o(attn_output)

        return output  # type: ignore[no-any-return]

    def _create_attention_mask(
        self,
        seq_lens: mx.array,
        max_seq_len: int,
    ) -> Optional[mx.array]:
        """Create attention mask for variable-length sequences.

        For global attention, we create a mask that prevents attending to
        padding positions beyond each sample's actual sequence length.

        Args:
            seq_lens: Tensor of shape [B] with actual sequence lengths
            max_seq_len: Maximum sequence length (L)

        Returns:
            Mask tensor of shape [B, 1, L, L] or None if all sequences are same length
        """
        batch_size = seq_lens.shape[0]

        # Check if all sequences have the same length (no padding)
        # In that case, we don't need a mask
        seq_lens_list = [int(seq_lens[i].item()) for i in range(batch_size)]
        if all(sl == max_seq_len for sl in seq_lens_list):
            return None

        # Create position indices
        positions = mx.arange(max_seq_len)  # [L]

        # Create mask: positions < seq_len for each sample
        # Shape: [B, L]
        masks = []
        for i in range(batch_size):
            seq_len = seq_lens_list[i]
            # For key positions: True if position < seq_len (valid), False otherwise (padding)
            mask_row = positions < seq_len  # [L]
            masks.append(mask_row)

        # Stack to [B, L] and expand for broadcasting
        key_mask = mx.stack(masks, axis=0)  # [B, L]

        # Expand to [B, 1, 1, L] for broadcasting with attention scores [B, H, L, L]
        # The mask indicates which key positions are valid (True = valid, False = mask out)
        key_mask = mx.expand_dims(key_mask, axis=(1, 2))  # [B, 1, 1, L]

        # For scaled_dot_product_attention, mask should be additive:
        # 0 for valid positions, -inf for masked positions
        # Convert boolean mask to float mask
        zeros = mx.zeros(key_mask.shape, dtype=mx.float32)
        neg_inf = mx.full(key_mask.shape, float('-inf'), dtype=mx.float32)
        mask = mx.where(key_mask, zeros, neg_inf)

        return mask
