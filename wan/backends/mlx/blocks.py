"""MLX implementation of WanAttentionBlock for WanModel.

This module provides an MLX-native implementation of the transformer block
to enable Metal-accelerated inference on Apple Silicon.

Key components:
- Self-attention with RoPE
- Cross-attention for text conditioning
- FFN with GELU(tanh) activation
- 6-way modulation from time embeddings
- Camera pose injection layers

Reference: wan/modules/model.py (WanAttentionBlock class)
"""

from typing import Any, Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .attention import WanCrossAttentionMLX, WanSelfAttentionMLX
from .norms import WanLayerNormMLX

__all__ = ['WanAttentionBlockMLX']


class WanAttentionBlockMLX(nn.Module):  # type: ignore[misc, name-defined]
    """Transformer attention block with FFN and camera injection for MLX.

    This is a direct port of WanAttentionBlock from wan/modules/model.py to MLX.
    Combines self-attention, cross-attention, and FFN with modulation from
    time embeddings and optional camera pose injection.

    Args:
        dim: Hidden dimension of the model (e.g., 2048)
        ffn_dim: FFN hidden dimension (e.g., 8192)
        num_heads: Number of attention heads (e.g., 16)
        window_size: Window size for local attention. (-1, -1) means global attention.
        qk_norm: If True, apply RMSNorm to query and key projections
        cross_attn_norm: If True, apply LayerNorm before cross-attention
        eps: Epsilon for normalization layers
    """

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        window_size: Tuple[int, int] = (-1, -1),
        qk_norm: bool = True,
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # Layer normalizations
        self.norm1 = WanLayerNormMLX(dim, eps=eps, elementwise_affine=False)
        self.norm2 = WanLayerNormMLX(dim, eps=eps, elementwise_affine=False)
        # norm3 is identity when cross_attn_norm=False, LayerNorm with affine when True
        self._use_norm3 = cross_attn_norm
        if cross_attn_norm:
            self.norm3 = WanLayerNormMLX(dim, eps=eps, elementwise_affine=True)

        # Self-attention and cross-attention
        self.self_attn = WanSelfAttentionMLX(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            qk_norm=qk_norm,
            eps=eps,
        )
        self.cross_attn = WanCrossAttentionMLX(
            dim=dim,
            num_heads=num_heads,
            window_size=(-1, -1),  # Cross-attention always global
            qk_norm=qk_norm,
            eps=eps,
        )

        # FFN: Linear -> GELU(tanh) -> Linear
        self.ffn_linear1: Any = nn.Linear(dim, ffn_dim)  # type: ignore[attr-defined]
        self.ffn_linear2: Any = nn.Linear(ffn_dim, dim)  # type: ignore[attr-defined]

        # Modulation parameter (1, 6, dim) - splits into 6 parts for different modulation points
        # Initialize with randn / sqrt(dim) to match PyTorch
        self.modulation = mx.random.normal((1, 6, dim)) / (dim ** 0.5)

        # Camera injection layers
        self.cam_injector_layer1: Any = nn.Linear(dim, dim)  # type: ignore[attr-defined]
        self.cam_injector_layer2: Any = nn.Linear(dim, dim)  # type: ignore[attr-defined]
        self.cam_scale_layer: Any = nn.Linear(dim, dim)  # type: ignore[attr-defined]
        self.cam_shift_layer: Any = nn.Linear(dim, dim)  # type: ignore[attr-defined]

    def _ffn(self, x: mx.array) -> mx.array:
        """Feed-forward network with GELU(tanh) activation.

        Args:
            x: Input tensor of shape [B, L, dim]

        Returns:
            Output tensor of shape [B, L, dim]
        """
        # Linear -> GELU(approximate='tanh') -> Linear
        x = self.ffn_linear1(x)
        # MLX's nn.gelu_approx uses tanh approximation
        x = nn.gelu_approx(x)  # type: ignore[attr-defined]
        x = self.ffn_linear2(x)
        return x

    def __call__(
        self,
        x: mx.array,
        e: mx.array,
        seq_lens: mx.array,
        grid_sizes: mx.array,
        freqs_cos: Tuple[mx.array, mx.array, mx.array],
        freqs_sin: Tuple[mx.array, mx.array, mx.array],
        context: mx.array,
        context_lens: Optional[mx.array],
        dit_cond_dict: Optional[Dict[str, mx.array]] = None,
    ) -> mx.array:
        """Forward pass of the attention block.

        Args:
            x: Hidden states of shape [B, L, C]
            e: Time embedding of shape [B, 1, 6, C] for modulation
            seq_lens: Tensor of shape [B] containing actual sequence lengths
            grid_sizes: Tensor of shape [B, 3] containing (F, H, W) grid dimensions
            freqs_cos: Tuple of 3 cosine frequency tensors for RoPE (frame, height, width)
            freqs_sin: Tuple of 3 sine frequency tensors for RoPE (frame, height, width)
            context: Text embeddings of shape [B, text_len, C]
            context_lens: Tensor of shape [B] containing actual context lengths
            dit_cond_dict: Optional dict containing camera embeddings:
                - 'c2ws_plucker_emb': Camera Plucker embeddings of shape [B, L, C]

        Returns:
            Output tensor of shape [B, L, C]
        """
        # Apply modulation
        # e shape: [B, 1, 6, C]
        # modulation shape: [1, 6, C]
        # Add modulation and expand: [B, 1, 6, C]
        e_mod = mx.expand_dims(self.modulation, axis=0) + e
        # Split into 6 parts along dim=2, each of shape [B, 1, 1, C]
        # We chunk along axis 2: [B, 1, 6, C] -> 6 x [B, 1, 1, C]
        e0 = e_mod[:, :, 0:1, :]  # [B, 1, 1, C]
        e1 = e_mod[:, :, 1:2, :]
        e2 = e_mod[:, :, 2:3, :]
        e3 = e_mod[:, :, 3:4, :]
        e4 = e_mod[:, :, 4:5, :]
        e5 = e_mod[:, :, 5:6, :]

        # Squeeze the chunk dimension: [B, 1, 1, C] -> [B, 1, C]
        e0 = mx.squeeze(e0, axis=2)
        e1 = mx.squeeze(e1, axis=2)
        e2 = mx.squeeze(e2, axis=2)
        e3 = mx.squeeze(e3, axis=2)
        e4 = mx.squeeze(e4, axis=2)
        e5 = mx.squeeze(e5, axis=2)

        # Self-attention with modulation
        # norm1(x) * (1 + e1) + e0
        x_normed = self.norm1(x).astype(mx.float32)
        x_mod = x_normed * (1.0 + e1) + e0
        y = self.self_attn(x_mod, seq_lens, grid_sizes, freqs_cos, freqs_sin)
        # x = x + y * e2
        x = x + y * e2

        # Camera injection (optional)
        if dit_cond_dict is not None and "c2ws_plucker_emb" in dit_cond_dict:
            c2ws_plucker_emb = dit_cond_dict["c2ws_plucker_emb"]
            # SiLU activation: x * sigmoid(x)
            c2ws_hidden = nn.silu(self.cam_injector_layer1(c2ws_plucker_emb))  # type: ignore[attr-defined]
            c2ws_hidden = self.cam_injector_layer2(c2ws_hidden)
            # Residual connection
            c2ws_hidden = c2ws_hidden + c2ws_plucker_emb
            # Scale and shift modulation
            cam_scale = self.cam_scale_layer(c2ws_hidden)
            cam_shift = self.cam_shift_layer(c2ws_hidden)
            x = (1.0 + cam_scale) * x + cam_shift

        # Cross-attention
        # Apply norm3 if cross_attn_norm=True, otherwise identity
        if self._use_norm3:
            x_for_cross = self.norm3(x)
        else:
            x_for_cross = x
        x = x + self.cross_attn(x_for_cross, context, context_lens)

        # FFN with modulation
        # norm2(x) * (1 + e4) + e3
        x_normed = self.norm2(x).astype(mx.float32)
        x_mod = x_normed * (1.0 + e4) + e3
        y = self._ffn(x_mod)
        # x = x + y * e5
        x = x + y * e5

        return x
