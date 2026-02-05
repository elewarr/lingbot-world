#!/usr/bin/env python3
"""Benchmark for MLX RoPE vectorization optimization.

Compares the old loop-based implementation vs the new vectorized implementation.
"""

import time

import mlx.core as mx
import numpy as np

from wan.backends.mlx.rope import (
    _build_freq_grid,
    _rope_apply_uniform,
    create_rope_freqs_mlx,
    rope_apply_mlx,
)


def rope_apply_mlx_original(
    x: mx.array,
    grid_sizes: mx.array,
    freqs_cos,
    freqs_sin,
) -> mx.array:
    """Original loop-based implementation for comparison."""
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
        x_pairs = x_i.reshape(seq_len, num_heads, -1, 2)
        x_re = x_pairs[..., 0]  # [seq_len, N, D//2]
        x_im = x_pairs[..., 1]  # [seq_len, N, D//2]

        # Frame frequencies
        cos_f = freqs_cos[0][:f]
        sin_f = freqs_sin[0][:f]
        cos_f = mx.expand_dims(cos_f, axis=(1, 2))
        sin_f = mx.expand_dims(sin_f, axis=(1, 2))
        cos_f = mx.broadcast_to(cos_f, (f, h, w, cos_f.shape[-1]))
        sin_f = mx.broadcast_to(sin_f, (f, h, w, sin_f.shape[-1]))

        # Height frequencies
        cos_h = freqs_cos[1][:h]
        sin_h = freqs_sin[1][:h]
        cos_h = mx.expand_dims(cos_h, axis=(0, 2))
        sin_h = mx.expand_dims(sin_h, axis=(0, 2))
        cos_h = mx.broadcast_to(cos_h, (f, h, w, cos_h.shape[-1]))
        sin_h = mx.broadcast_to(sin_h, (f, h, w, sin_h.shape[-1]))

        # Width frequencies
        cos_w = freqs_cos[2][:w]
        sin_w = freqs_sin[2][:w]
        cos_w = mx.expand_dims(cos_w, axis=(0, 1))
        sin_w = mx.expand_dims(sin_w, axis=(0, 1))
        cos_w = mx.broadcast_to(cos_w, (f, h, w, cos_w.shape[-1]))
        sin_w = mx.broadcast_to(sin_w, (f, h, w, sin_w.shape[-1]))

        # Concatenate
        cos_grid = mx.concatenate([cos_f, cos_h, cos_w], axis=-1)
        sin_grid = mx.concatenate([sin_f, sin_h, sin_w], axis=-1)

        # Reshape for broadcasting
        cos_grid = cos_grid.reshape(seq_len, 1, -1)
        sin_grid = sin_grid.reshape(seq_len, 1, -1)

        # Apply rotation
        out_re = x_re * cos_grid - x_im * sin_grid
        out_im = x_re * sin_grid + x_im * cos_grid

        # Interleave
        x_out = mx.stack([out_re, out_im], axis=-1)
        x_out = x_out.reshape(seq_len, num_heads, -1)

        # Padding
        if seq_len < seq_len_max:
            padding = x[i, seq_len:]
            x_out = mx.concatenate([x_out, padding], axis=0)

        output.append(x_out)

    return mx.stack(output, axis=0).astype(mx.float32)


def bench_rope_apply(
    impl_fn,
    batch_size: int,
    f: int,
    h: int,
    w: int,
    num_heads: int = 16,
    head_dim: int = 128,
    warmup_iters: int = 5,
    bench_iters: int = 50,
    variable_grids: bool = False,
):
    """Benchmark rope_apply_mlx.

    Args:
        impl_fn: The RoPE implementation function to benchmark
        batch_size: Number of samples in batch
        f, h, w: Grid dimensions
        num_heads: Number of attention heads
        head_dim: Dimension per head
        warmup_iters: Number of warmup iterations
        bench_iters: Number of benchmark iterations
        variable_grids: If True, use different grid sizes per sample
    """
    seq_len = f * h * w
    max_seq_len = seq_len + 64  # Some padding

    # Create input
    np.random.seed(42)
    x_np = np.random.randn(batch_size, max_seq_len, num_heads, head_dim).astype(
        np.float32
    )
    x = mx.array(x_np)

    # Create grid sizes
    if variable_grids and batch_size > 1:
        # Different grid sizes per sample
        grid_list = []
        for i in range(batch_size):
            # Vary dimensions slightly
            fi = max(1, f - i % 3)
            hi = max(1, h + i % 2)
            wi = w
            grid_list.append([fi, hi, wi])
        grid_sizes = mx.array(np.array(grid_list, dtype=np.int64))
    else:
        grid_sizes = mx.array(np.array([[f, h, w]] * batch_size, dtype=np.int64))

    # Create frequencies
    cos_tuple, sin_tuple = create_rope_freqs_mlx(1024, head_dim)

    # Warmup
    for _ in range(warmup_iters):
        out = impl_fn(x, grid_sizes, cos_tuple, sin_tuple)
        mx.eval(out)

    # Benchmark
    times = []
    for _ in range(bench_iters):
        start = time.perf_counter()
        out = impl_fn(x, grid_sizes, cos_tuple, sin_tuple)
        mx.eval(out)
        end = time.perf_counter()
        times.append(end - start)

    times = np.array(times)
    return {
        "mean_ms": times.mean() * 1000,
        "std_ms": times.std() * 1000,
        "min_ms": times.min() * 1000,
        "max_ms": times.max() * 1000,
        "positions": batch_size * seq_len,
    }


def main():
    print("=" * 70)
    print("MLX RoPE Vectorization Benchmark")
    print("=" * 70)
    print()

    # Test configurations
    configs = [
        # (batch_size, f, h, w, description)
        (1, 11, 15, 26, "WanModel 480p (B=1, 4290 pos)"),
        (1, 21, 30, 52, "WanModel 720p (B=1, 32760 pos)"),
        (1, 8, 8, 16, "Small video (B=1, 1024 pos)"),
        (2, 8, 6, 8, "Batch uniform (B=2, 384 pos each)"),
        (4, 8, 6, 8, "Batch uniform (B=4, 384 pos each)"),
    ]

    print("Configuration: num_heads=16, head_dim=128")
    print("Warmup: 5 iters, Benchmark: 50 iters")
    print()

    print("-" * 70)
    print("COMPARISON: Original (loop) vs Vectorized")
    print("-" * 70)
    print()

    for batch_size, f, h, w, desc in configs:
        result_orig = bench_rope_apply(rope_apply_mlx_original, batch_size, f, h, w)
        result_new = bench_rope_apply(rope_apply_mlx, batch_size, f, h, w)
        seq_len = f * h * w

        speedup = result_orig["mean_ms"] / result_new["mean_ms"]

        print(f"{desc}")
        print(f"  Grid: ({f}, {h}, {w}) = {seq_len} positions")
        print(
            f"  Original: {result_orig['mean_ms']:.3f} +/- {result_orig['std_ms']:.3f} ms"
        )
        print(
            f"  Vectorized: {result_new['mean_ms']:.3f} +/- {result_new['std_ms']:.3f} ms"
        )
        print(f"  Speedup: {speedup:.2f}x")
        print()

    # Variable grid sizes test
    print("-" * 70)
    print("Variable grid sizes (fallback path):")
    print("-" * 70)
    result_orig = bench_rope_apply(
        rope_apply_mlx_original, 4, 8, 6, 8, variable_grids=True
    )
    result_new = bench_rope_apply(rope_apply_mlx, 4, 8, 6, 8, variable_grids=True)
    speedup = result_orig["mean_ms"] / result_new["mean_ms"]
    print(
        f"  Original: {result_orig['mean_ms']:.3f} +/- {result_orig['std_ms']:.3f} ms"
    )
    print(
        f"  Vectorized: {result_new['mean_ms']:.3f} +/- {result_new['std_ms']:.3f} ms"
    )
    print(f"  Speedup: {speedup:.2f}x")
    print()

    print("=" * 70)
    print("Benchmark complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
