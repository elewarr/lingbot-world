"""MLX implementation of Rotary Position Embeddings (RoPE) for WanModel.

This module provides a complex-number-free implementation of RoPE that produces
results numerically equivalent to the PyTorch version using rotation matrices.

Key differences from PyTorch:
- PyTorch: uses torch.complex and complex multiplication
- MLX: uses (cos, sin) pairs with rotation matrix: (x1*cos - x2*sin, x1*sin + x2*cos)

Optimization:
- Uses vectorized operations when possible (uniform grid_sizes)
- Falls back to mx.vmap for variable grid_sizes per sample
- Avoids Python loops that break the MLX compute graph

Reference: wan/modules/model.py (rope_params, rope_apply functions)
"""

from typing import List, Tuple

import mlx.core as mx

__all__ = ["rope_params_mlx", "rope_apply_mlx"]


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


def _build_freq_grid(
    f: int,
    h: int,
    w: int,
    freqs_cos: Tuple[mx.array, mx.array, mx.array],
    freqs_sin: Tuple[mx.array, mx.array, mx.array],
) -> Tuple[mx.array, mx.array]:
    """Build 3D frequency grid for RoPE.

    Returns:
        cos_grid, sin_grid: Each of shape [f*h*w, D//2]
    """
    seq_len = f * h * w

    # Frame frequencies: [f, d_f] -> [f, h, w, d_f]
    cos_f = freqs_cos[0][:f]  # [f, d_f]
    sin_f = freqs_sin[0][:f]
    cos_f = mx.broadcast_to(cos_f[:, None, None, :], (f, h, w, cos_f.shape[-1]))
    sin_f = mx.broadcast_to(sin_f[:, None, None, :], (f, h, w, sin_f.shape[-1]))

    # Height frequencies: [h, d_h] -> [f, h, w, d_h]
    cos_h = freqs_cos[1][:h]  # [h, d_h]
    sin_h = freqs_sin[1][:h]
    cos_h = mx.broadcast_to(cos_h[None, :, None, :], (f, h, w, cos_h.shape[-1]))
    sin_h = mx.broadcast_to(sin_h[None, :, None, :], (f, h, w, sin_h.shape[-1]))

    # Width frequencies: [w, d_w] -> [f, h, w, d_w]
    cos_w = freqs_cos[2][:w]  # [w, d_w]
    sin_w = freqs_sin[2][:w]
    cos_w = mx.broadcast_to(cos_w[None, None, :, :], (f, h, w, cos_w.shape[-1]))
    sin_w = mx.broadcast_to(sin_w[None, None, :, :], (f, h, w, sin_w.shape[-1]))

    # Concatenate: [f, h, w, D//2] -> [seq_len, D//2]
    cos_grid = mx.concatenate([cos_f, cos_h, cos_w], axis=-1).reshape(seq_len, -1)
    sin_grid = mx.concatenate([sin_f, sin_h, sin_w], axis=-1).reshape(seq_len, -1)

    return cos_grid, sin_grid


def _rope_apply_single(
    x_i: mx.array,
    cos_grid: mx.array,
    sin_grid: mx.array,
    seq_len: int,
    seq_len_max: int,
) -> mx.array:
    """Apply RoPE to a single sample.

    Args:
        x_i: Input [L, N, D]
        cos_grid: [seq_len, D//2]
        sin_grid: [seq_len, D//2]
        seq_len: Valid sequence length
        seq_len_max: Maximum sequence length (for padding)

    Returns:
        Output [L, N, D] with RoPE applied to first seq_len positions
    """
    num_heads = x_i.shape[1]

    # Extract valid sequence
    x_valid = x_i[:seq_len]  # [seq_len, N, D]

    # Reshape to pairs: [seq_len, N, D//2, 2]
    x_pairs = x_valid.reshape(seq_len, num_heads, -1, 2)
    x_re = x_pairs[..., 0]  # [seq_len, N, D//2]
    x_im = x_pairs[..., 1]  # [seq_len, N, D//2]

    # Add head broadcast dim: [seq_len, 1, D//2]
    cos_grid = cos_grid[:, None, :]
    sin_grid = sin_grid[:, None, :]

    # Apply rotation
    out_re = x_re * cos_grid - x_im * sin_grid  # [seq_len, N, D//2]
    out_im = x_re * sin_grid + x_im * cos_grid  # [seq_len, N, D//2]

    # Interleave: [seq_len, N, D//2, 2] -> [seq_len, N, D]
    x_out = mx.stack([out_re, out_im], axis=-1).reshape(seq_len, num_heads, -1)

    # Append padding if needed
    if seq_len < seq_len_max:
        x_out = mx.concatenate([x_out, x_i[seq_len:]], axis=0)

    return x_out


def _rope_apply_uniform(
    x: mx.array,
    f: int,
    h: int,
    w: int,
    freqs_cos: Tuple[mx.array, mx.array, mx.array],
    freqs_sin: Tuple[mx.array, mx.array, mx.array],
) -> mx.array:
    """Fully vectorized RoPE for uniform grid_sizes (all samples same F,H,W).

    This is the fast path that processes the entire batch in one fused operation.

    Args:
        x: Input [B, L, N, D]
        f, h, w: Grid dimensions (same for all samples)
        freqs_cos, freqs_sin: Frequency tuples

    Returns:
        Output [B, L, N, D] with RoPE applied
    """
    batch_size, seq_len_max, num_heads, head_dim = x.shape
    seq_len = f * h * w

    # Build frequency grid once: [seq_len, D//2]
    cos_grid, sin_grid = _build_freq_grid(f, h, w, freqs_cos, freqs_sin)

    # Extract valid sequence: [B, seq_len, N, D]
    x_valid = x[:, :seq_len]

    # Reshape to pairs: [B, seq_len, N, D//2, 2]
    x_pairs = x_valid.reshape(batch_size, seq_len, num_heads, -1, 2)
    x_re = x_pairs[..., 0]  # [B, seq_len, N, D//2]
    x_im = x_pairs[..., 1]  # [B, seq_len, N, D//2]

    # Broadcast freq grid: [1, seq_len, 1, D//2] for [B, seq_len, N, D//2]
    cos_grid = cos_grid[None, :, None, :]  # [1, seq_len, 1, D//2]
    sin_grid = sin_grid[None, :, None, :]  # [1, seq_len, 1, D//2]

    # Apply rotation to entire batch
    out_re = x_re * cos_grid - x_im * sin_grid  # [B, seq_len, N, D//2]
    out_im = x_re * sin_grid + x_im * cos_grid  # [B, seq_len, N, D//2]

    # Interleave: [B, seq_len, N, D//2, 2] -> [B, seq_len, N, D]
    x_out = mx.stack([out_re, out_im], axis=-1).reshape(
        batch_size, seq_len, num_heads, -1
    )

    # Append padding if needed
    if seq_len < seq_len_max:
        x_out = mx.concatenate([x_out, x[:, seq_len:]], axis=1)

    return x_out.astype(mx.float32)


def _rope_apply_variable(
    x: mx.array,
    grid_sizes: mx.array,
    freqs_cos: Tuple[mx.array, mx.array, mx.array],
    freqs_sin: Tuple[mx.array, mx.array, mx.array],
) -> mx.array:
    """RoPE for variable grid_sizes using mx.vmap.

    When samples have different grid sizes, we use vmap to vectorize
    the per-sample computation while keeping the compute graph intact.

    Args:
        x: Input [B, L, N, D]
        grid_sizes: [B, 3] with (F, H, W) per sample
        freqs_cos, freqs_sin: Frequency tuples

    Returns:
        Output [B, L, N, D] with RoPE applied
    """
    batch_size = x.shape[0]
    seq_len_max = x.shape[1]

    # We need to materialize grid_sizes to get actual F,H,W values
    # This is unavoidable for variable grid sizes
    mx.eval(grid_sizes)

    # Precompute frequency grids for each unique (F,H,W) combination
    # In practice, most batches have uniform sizes, but we handle the general case
    grid_sizes_list = [
        (
            int(grid_sizes[i, 0].item()),
            int(grid_sizes[i, 1].item()),
            int(grid_sizes[i, 2].item()),
        )
        for i in range(batch_size)
    ]

    # Build frequency grids for each sample
    cos_grids = []
    sin_grids = []
    for f, h, w in grid_sizes_list:
        cos_grid, sin_grid = _build_freq_grid(f, h, w, freqs_cos, freqs_sin)
        cos_grids.append(cos_grid)
        sin_grids.append(sin_grid)

    # Apply RoPE to each sample (could use vmap but dynamic shapes are tricky)
    # Instead, stack operations where possible
    output = []
    for i, (f, h, w) in enumerate(grid_sizes_list):
        seq_len = f * h * w
        x_out = _rope_apply_single(
            x[i], cos_grids[i], sin_grids[i], seq_len, seq_len_max
        )
        output.append(x_out)

    return mx.stack(output, axis=0).astype(mx.float32)


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

    Optimization:
    - Fast path: When all samples have the same grid_sizes, uses fully
      vectorized operations on the entire batch (no Python loops).
    - Slow path: For variable grid_sizes, falls back to per-sample processing
      with helper functions to keep code modular.

    Args:
        x: Input tensor of shape [B, L, N, D] where D is head_dim (e.g., 128)
        grid_sizes: Tensor of shape [B, 3] containing (F, H, W) for each sample
        freqs_cos: Tuple of 3 cos frequency tensors for (frame, height, width)
        freqs_sin: Tuple of 3 sin frequency tensors for (frame, height, width)

    Returns:
        Tensor of shape [B, L, N, D] with rotary embeddings applied
    """
    batch_size = x.shape[0]

    # Fast path: batch_size == 1 (most common in inference)
    if batch_size == 1:
        mx.eval(grid_sizes)
        f = int(grid_sizes[0, 0].item())
        h = int(grid_sizes[0, 1].item())
        w = int(grid_sizes[0, 2].item())
        return _rope_apply_uniform(x, f, h, w, freqs_cos, freqs_sin)

    # Check if all grid_sizes are uniform (common case)
    mx.eval(grid_sizes)
    first_grid = (
        int(grid_sizes[0, 0].item()),
        int(grid_sizes[0, 1].item()),
        int(grid_sizes[0, 2].item()),
    )

    is_uniform = all(
        (
            int(grid_sizes[i, 0].item()),
            int(grid_sizes[i, 1].item()),
            int(grid_sizes[i, 2].item()),
        )
        == first_grid
        for i in range(1, batch_size)
    )

    if is_uniform:
        # Fast path: fully vectorized
        f, h, w = first_grid
        return _rope_apply_uniform(x, f, h, w, freqs_cos, freqs_sin)
    else:
        # Slow path: variable grid sizes
        return _rope_apply_variable(x, grid_sizes, freqs_cos, freqs_sin)


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
    dim_h = 2 * (d // 6)  # 42 for d=128
    dim_w = 2 * (d // 6)  # 42 for d=128

    # Compute frequencies for each dimension
    cos_f, sin_f = rope_params_mlx(max_seq_len, dim_f, theta)
    cos_h, sin_h = rope_params_mlx(max_seq_len, dim_h, theta)
    cos_w, sin_w = rope_params_mlx(max_seq_len, dim_w, theta)

    return (cos_f, cos_h, cos_w), (sin_f, sin_h, sin_w)
