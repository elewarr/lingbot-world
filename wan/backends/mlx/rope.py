"""MLX implementation of Rotary Position Embeddings (RoPE) for WanModel.

This module provides a complex-number-free implementation of RoPE that produces
results numerically equivalent to the PyTorch version using rotation matrices.

Key differences from PyTorch:
- PyTorch: uses torch.complex and complex multiplication
- MLX: uses (cos, sin) pairs with rotation matrix: (x1*cos - x2*sin, x1*sin + x2*cos)

Reference: wan/modules/model.py (rope_params, rope_apply functions)
"""

from typing import Tuple, List

import mlx.core as mx

__all__ = ['rope_params_mlx', 'rope_apply_mlx']


def rope_params_mlx(
    max_seq_len: int,
    dim: int,
    theta: float = 10000.0,
) -> Tuple[mx.array, mx.array]:
    """Compute rotary position embedding parameters (cos, sin).

    This is equivalent to the PyTorch rope_params but returns real-valued
    (cos, sin) pairs instead of complex numbers.

    Args:
        max_seq_len: Maximum sequence length for positional encoding
        dim: Dimension of the embeddings (must be even)
        theta: Base for the exponential frequency scaling

    Returns:
        Tuple of (freqs_cos, freqs_sin), each with shape [max_seq_len, dim//2]
    """
    assert dim % 2 == 0, f"dim must be even, got {dim}"

    # Compute frequencies: 1 / theta^(2i/dim) for i in [0, 1, ..., dim//2 - 1]
    # Equivalent to torch.arange(0, dim, 2).div(dim)
    half_dim = dim // 2
    freq_exponents = mx.arange(0, dim, 2, dtype=mx.float32) / dim
    inv_freqs = 1.0 / mx.power(theta, freq_exponents)  # [dim//2]

    # Compute angles: position * frequency for all positions and frequencies
    positions = mx.arange(max_seq_len, dtype=mx.float32)  # [max_seq_len]
    angles = mx.outer(positions, inv_freqs)  # [max_seq_len, dim//2]

    # Return cos and sin of the angles
    freqs_cos = mx.cos(angles)  # [max_seq_len, dim//2]
    freqs_sin = mx.sin(angles)  # [max_seq_len, dim//2]

    return freqs_cos, freqs_sin


def rope_apply_mlx(
    x: mx.array,
    grid_sizes: mx.array,
    freqs_cos: Tuple[mx.array, mx.array, mx.array],
    freqs_sin: Tuple[mx.array, mx.array, mx.array],
) -> mx.array:
    """Apply rotary position embeddings using rotation matrices.

    This is equivalent to the PyTorch rope_apply but uses real-valued
    rotation instead of complex multiplication.

    The key insight is that complex multiplication (a+bi)(c+di) = (ac-bd) + i(ad+bc)
    can be rewritten as a rotation matrix applied to [a, b]:
        [a', b'] = [a*c - b*d, a*d + b*c]

    In our case, freqs are e^(i*angle) = cos(angle) + i*sin(angle),
    so c=cos, d=sin, and we get:
        [x_re', x_im'] = [x_re*cos - x_im*sin, x_re*sin + x_im*cos]

    Args:
        x: Input tensor of shape [B, L, N, D] where D is head_dim (e.g., 128)
        grid_sizes: Tensor of shape [B, 3] containing (F, H, W) for each sample
        freqs_cos: Tuple of 3 cos frequency tensors for (frame, height, width)
        freqs_sin: Tuple of 3 sin frequency tensors for (frame, height, width)

    Returns:
        Tensor of shape [B, L, N, D] with rotary embeddings applied
    """
    batch_size = x.shape[0]
    seq_len_max = x.shape[1]
    num_heads = x.shape[2]
    head_dim = x.shape[3]

    # Process each sample in the batch
    output = []
    for i in range(batch_size):
        # Get grid dimensions for this sample
        f = int(grid_sizes[i, 0].item())
        h = int(grid_sizes[i, 1].item())
        w = int(grid_sizes[i, 2].item())
        seq_len = f * h * w

        # Extract the valid sequence for this sample: [seq_len, N, D]
        x_i = x[i, :seq_len]

        # Reshape to group consecutive pairs as (real, imag)
        # PyTorch does: x.reshape(seq_len, n, -1, 2) then view_as_complex
        # So consecutive elements [0,1], [2,3], etc. form complex pairs
        # Shape: [seq_len, N, D//2, 2]
        x_pairs = x_i.reshape(seq_len, num_heads, -1, 2)
        x_re = x_pairs[..., 0]  # [seq_len, N, D//2]
        x_im = x_pairs[..., 1]  # [seq_len, N, D//2]

        # Build the frequency grid for 3D positions (F, H, W)
        # Each position encodes its frame, height, and width coordinates
        # freqs_cos/sin are tuples: (frame_freqs, height_freqs, width_freqs)

        # Frame frequencies: expand [f, d_f] to [f, h, w, d_f]
        cos_f = freqs_cos[0][:f]  # [f, d_f]
        sin_f = freqs_sin[0][:f]  # [f, d_f]
        cos_f = mx.expand_dims(cos_f, axis=(1, 2))  # [f, 1, 1, d_f]
        sin_f = mx.expand_dims(sin_f, axis=(1, 2))  # [f, 1, 1, d_f]
        cos_f = mx.broadcast_to(cos_f, (f, h, w, cos_f.shape[-1]))  # [f, h, w, d_f]
        sin_f = mx.broadcast_to(sin_f, (f, h, w, sin_f.shape[-1]))  # [f, h, w, d_f]

        # Height frequencies: expand [h, d_h] to [f, h, w, d_h]
        cos_h = freqs_cos[1][:h]  # [h, d_h]
        sin_h = freqs_sin[1][:h]  # [h, d_h]
        cos_h = mx.expand_dims(cos_h, axis=(0, 2))  # [1, h, 1, d_h]
        sin_h = mx.expand_dims(sin_h, axis=(0, 2))  # [1, h, 1, d_h]
        cos_h = mx.broadcast_to(cos_h, (f, h, w, cos_h.shape[-1]))  # [f, h, w, d_h]
        sin_h = mx.broadcast_to(sin_h, (f, h, w, sin_h.shape[-1]))  # [f, h, w, d_h]

        # Width frequencies: expand [w, d_w] to [f, h, w, d_w]
        cos_w = freqs_cos[2][:w]  # [w, d_w]
        sin_w = freqs_sin[2][:w]  # [w, d_w]
        cos_w = mx.expand_dims(cos_w, axis=(0, 1))  # [1, 1, w, d_w]
        sin_w = mx.expand_dims(sin_w, axis=(0, 1))  # [1, 1, w, d_w]
        cos_w = mx.broadcast_to(cos_w, (f, h, w, cos_w.shape[-1]))  # [f, h, w, d_w]
        sin_w = mx.broadcast_to(sin_w, (f, h, w, sin_w.shape[-1]))  # [f, h, w, d_w]

        # Concatenate along the frequency dimension
        # Total dim: d_f + d_h + d_w = D//2
        cos_grid = mx.concatenate([cos_f, cos_h, cos_w], axis=-1)  # [f, h, w, D//2]
        sin_grid = mx.concatenate([sin_f, sin_h, sin_w], axis=-1)  # [f, h, w, D//2]

        # Reshape to [seq_len, 1, D//2] for broadcasting with [seq_len, N, D//2]
        cos_grid = cos_grid.reshape(seq_len, 1, -1)  # [seq_len, 1, D//2]
        sin_grid = sin_grid.reshape(seq_len, 1, -1)  # [seq_len, 1, D//2]

        # Apply rotation: (x_re*cos - x_im*sin, x_re*sin + x_im*cos)
        out_re = x_re * cos_grid - x_im * sin_grid  # [seq_len, N, D//2]
        out_im = x_re * sin_grid + x_im * cos_grid  # [seq_len, N, D//2]

        # Interleave back to original layout
        # Stack to [seq_len, N, D//2, 2] then reshape to [seq_len, N, D]
        x_out = mx.stack([out_re, out_im], axis=-1)  # [seq_len, N, D//2, 2]
        x_out = x_out.reshape(seq_len, num_heads, -1)  # [seq_len, N, D]

        # Concatenate with padding (positions beyond seq_len)
        if seq_len < seq_len_max:
            padding = x[i, seq_len:]  # [seq_len_max - seq_len, N, D]
            x_out = mx.concatenate([x_out, padding], axis=0)

        output.append(x_out)

    # Stack all samples back to batch
    return mx.stack(output, axis=0).astype(mx.float32)


def split_freqs_for_3d_rope(
    freqs_cos: mx.array,
    freqs_sin: mx.array,
    head_dim: int = 128,
) -> Tuple[Tuple[mx.array, mx.array, mx.array], Tuple[mx.array, mx.array, mx.array]]:
    """Split frequency tensors for 3D RoPE (frame, height, width).

    In WanModel, the head dimension is split into 3 parts:
    - Frames: head_dim//2 - 2*(head_dim//2//3) dimensions (typically 22)
    - Height: head_dim//2//3 dimensions (typically 21)
    - Width: head_dim//2//3 dimensions (typically 21)

    This matches the PyTorch implementation:
        c = head_dim // 2  # 64
        freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    Args:
        freqs_cos: Full cosine frequencies [max_seq_len, head_dim//2]
        freqs_sin: Full sine frequencies [max_seq_len, head_dim//2]
        head_dim: The head dimension (default 128)

    Returns:
        Tuple of (cos_tuple, sin_tuple) where each contains 3 tensors
        for (frame, height, width) frequencies
    """
    c = head_dim // 2  # 64 for head_dim=128
    split_sizes = [c - 2 * (c // 3), c // 3, c // 3]  # [22, 21, 21] for c=64

    # Split along the last dimension
    # cumsum for split indices
    idx1 = split_sizes[0]
    idx2 = idx1 + split_sizes[1]

    cos_f = freqs_cos[:, :idx1]
    cos_h = freqs_cos[:, idx1:idx2]
    cos_w = freqs_cos[:, idx2:]

    sin_f = freqs_sin[:, :idx1]
    sin_h = freqs_sin[:, idx1:idx2]
    sin_w = freqs_sin[:, idx2:]

    return (cos_f, cos_h, cos_w), (sin_f, sin_h, sin_w)


def create_rope_freqs_mlx(
    max_seq_len: int = 1024,
    head_dim: int = 128,
    theta: float = 10000.0,
) -> Tuple[Tuple[mx.array, mx.array, mx.array], Tuple[mx.array, mx.array, mx.array]]:
    """Create RoPE frequency tensors ready for use with rope_apply_mlx.

    This is a convenience function that:
    1. Computes the base frequencies for each of the 3 dimensions
    2. Returns them in the split format expected by rope_apply_mlx

    The WanModel uses different frequency dimensions for frame, height, width:
    - Frame: d - 4*(d//6) where d = head_dim
    - Height: 2*(d//6)
    - Width: 2*(d//6)

    Args:
        max_seq_len: Maximum sequence length (default 1024)
        head_dim: Head dimension (default 128)
        theta: Base for exponential scaling (default 10000)

    Returns:
        Tuple of (cos_tuple, sin_tuple) ready for rope_apply_mlx
    """
    d = head_dim
    # These match the PyTorch rope_params calls in WanModel.__init__
    dim_f = d - 4 * (d // 6)  # 44 for d=128
    dim_h = 2 * (d // 6)      # 42 for d=128
    dim_w = 2 * (d // 6)      # 42 for d=128

    # Compute frequencies for each dimension
    cos_f, sin_f = rope_params_mlx(max_seq_len, dim_f, theta)
    cos_h, sin_h = rope_params_mlx(max_seq_len, dim_h, theta)
    cos_w, sin_w = rope_params_mlx(max_seq_len, dim_w, theta)

    return (cos_f, cos_h, cos_w), (sin_f, sin_h, sin_w)
