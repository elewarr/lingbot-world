"""Unit tests for MLX RoPE (Rotary Position Embeddings) implementation.

Tests that MLX implementation matches PyTorch reference within tolerance.
The key challenge is that PyTorch uses complex numbers while MLX uses
real-valued rotation matrices.
"""

import numpy as np
import pytest

# Import frameworks
import mlx.core as mx
import torch

# Import MLX implementations to test
from wan.backends.mlx.rope import (
    rope_params_mlx,
    rope_apply_mlx,
    split_freqs_for_3d_rope,
    create_rope_freqs_mlx,
)

# Import PyTorch reference
from wan.modules.model import rope_params, rope_apply


class TestRopeParamsMLX:
    """Tests for rope_params_mlx comparing against PyTorch rope_params."""

    def test_basic_shape(self):
        """Test that MLX rope_params produces correct output shapes."""
        max_seq_len = 1024
        dim = 128

        cos, sin = rope_params_mlx(max_seq_len, dim)
        mx.eval(cos)
        mx.eval(sin)

        # Each should be [max_seq_len, dim//2]
        assert cos.shape == (max_seq_len, dim // 2)
        assert sin.shape == (max_seq_len, dim // 2)

    def test_pytorch_equivalence(self):
        """Test that MLX cos/sin match PyTorch complex polar decomposition."""
        max_seq_len = 256
        dim = 128

        # PyTorch reference - returns complex numbers
        torch_freqs = rope_params(max_seq_len, dim)  # [max_seq_len, dim//2] complex

        # MLX implementation - returns cos, sin
        mlx_cos, mlx_sin = rope_params_mlx(max_seq_len, dim)
        mx.eval(mlx_cos)
        mx.eval(mlx_sin)

        # Extract real and imaginary parts from PyTorch complex
        # torch.polar(ones, angles) = cos(angles) + i*sin(angles)
        torch_cos = torch_freqs.real.numpy()
        torch_sin = torch_freqs.imag.numpy()

        mlx_cos_np = np.array(mlx_cos)
        mlx_sin_np = np.array(mlx_sin)

        # Compare with tolerance (looser due to float32 vs float64 precision)
        np.testing.assert_allclose(
            mlx_cos_np,
            torch_cos,
            rtol=1e-4,
            atol=1e-4,
            err_msg="Cosine frequencies don't match PyTorch"
        )
        np.testing.assert_allclose(
            mlx_sin_np,
            torch_sin,
            rtol=1e-4,
            atol=1e-4,
            err_msg="Sine frequencies don't match PyTorch"
        )

    def test_various_dimensions(self):
        """Test correctness across various dimension sizes."""
        test_cases = [
            (256, 64),
            (512, 128),
            (1024, 128),
            (1024, 256),
        ]

        for max_seq_len, dim in test_cases:
            torch_freqs = rope_params(max_seq_len, dim)
            mlx_cos, mlx_sin = rope_params_mlx(max_seq_len, dim)
            mx.eval(mlx_cos)
            mx.eval(mlx_sin)

            np.testing.assert_allclose(
                np.array(mlx_cos),
                torch_freqs.real.numpy(),
                rtol=1e-4,
                atol=1e-4,
                err_msg=f"Mismatch for max_seq_len={max_seq_len}, dim={dim}"
            )

    def test_custom_theta(self):
        """Test that custom theta parameter works correctly."""
        max_seq_len = 256
        dim = 64
        theta = 5000.0

        # PyTorch with custom theta
        torch_freqs = rope_params(max_seq_len, dim, theta=theta)

        # MLX with custom theta
        mlx_cos, mlx_sin = rope_params_mlx(max_seq_len, dim, theta=theta)
        mx.eval(mlx_cos)
        mx.eval(mlx_sin)

        np.testing.assert_allclose(
            np.array(mlx_cos),
            torch_freqs.real.numpy(),
            rtol=1e-4,
            atol=1e-4,
        )

    def test_dim_must_be_even(self):
        """Test that odd dimensions raise an assertion error."""
        with pytest.raises(AssertionError):
            rope_params_mlx(256, 127)


class TestSplitFreqsFor3DRope:
    """Tests for split_freqs_for_3d_rope utility function."""

    def test_split_sizes_default(self):
        """Test that split sizes match PyTorch for head_dim=128."""
        max_seq_len = 256
        head_dim = 128
        c = head_dim // 2  # 64

        # Create full frequencies
        cos, sin = rope_params_mlx(max_seq_len, head_dim)
        mx.eval(cos)
        mx.eval(sin)

        # Split
        cos_tuple, sin_tuple = split_freqs_for_3d_rope(cos, sin, head_dim)

        # Check shapes: [c - 2*(c//3), c//3, c//3] = [22, 21, 21] for c=64
        expected_sizes = [c - 2 * (c // 3), c // 3, c // 3]  # [22, 21, 21]

        assert cos_tuple[0].shape == (max_seq_len, expected_sizes[0])
        assert cos_tuple[1].shape == (max_seq_len, expected_sizes[1])
        assert cos_tuple[2].shape == (max_seq_len, expected_sizes[2])

        # Verify total size
        total_split = sum(c.shape[1] for c in cos_tuple)
        assert total_split == c  # Should sum to head_dim//2

    def test_split_matches_pytorch(self):
        """Test that split produces same results as PyTorch split."""
        max_seq_len = 256
        head_dim = 128
        c = head_dim // 2

        # PyTorch
        torch_freqs = rope_params(max_seq_len, head_dim)
        torch_split = torch_freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

        # MLX
        cos, sin = rope_params_mlx(max_seq_len, head_dim)
        cos_tuple, sin_tuple = split_freqs_for_3d_rope(cos, sin, head_dim)
        mx.eval(cos_tuple[0])
        mx.eval(cos_tuple[1])
        mx.eval(cos_tuple[2])

        # Compare each split (looser tolerance for precision differences)
        for i in range(3):
            np.testing.assert_allclose(
                np.array(cos_tuple[i]),
                torch_split[i].real.numpy(),
                rtol=1e-4,
                atol=1e-4,
                err_msg=f"Split {i} cos mismatch"
            )
            np.testing.assert_allclose(
                np.array(sin_tuple[i]),
                torch_split[i].imag.numpy(),
                rtol=1e-4,
                atol=1e-4,
                err_msg=f"Split {i} sin mismatch"
            )


class TestCreateRopeFreqsMLX:
    """Tests for create_rope_freqs_mlx convenience function."""

    def test_output_shapes(self):
        """Test that output shapes match WanModel conventions."""
        max_seq_len = 1024
        head_dim = 128
        d = head_dim

        cos_tuple, sin_tuple = create_rope_freqs_mlx(max_seq_len, head_dim)

        # Expected dimensions per WanModel
        dim_f = d - 4 * (d // 6)  # 44 for d=128
        dim_h = 2 * (d // 6)      # 42 for d=128
        dim_w = 2 * (d // 6)      # 42 for d=128

        # Each element is [max_seq_len, dim//2]
        assert cos_tuple[0].shape == (max_seq_len, dim_f // 2)  # 22
        assert cos_tuple[1].shape == (max_seq_len, dim_h // 2)  # 21
        assert cos_tuple[2].shape == (max_seq_len, dim_w // 2)  # 21

        # Total should be head_dim // 2
        total = sum(c.shape[1] for c in cos_tuple)
        assert total == head_dim // 2

    def test_matches_wanmodel_freqs_buffer(self):
        """Test that output matches WanModel's self.freqs buffer decomposition."""
        max_seq_len = 1024
        head_dim = 128
        d = head_dim

        # WanModel creates freqs like this:
        # self.freqs = torch.cat([
        #     rope_params(1024, d - 4 * (d // 6)),
        #     rope_params(1024, 2 * (d // 6)),
        #     rope_params(1024, 2 * (d // 6))
        # ], dim=1)

        # PyTorch reference
        torch_freqs_f = rope_params(max_seq_len, d - 4 * (d // 6))
        torch_freqs_h = rope_params(max_seq_len, 2 * (d // 6))
        torch_freqs_w = rope_params(max_seq_len, 2 * (d // 6))

        # MLX implementation
        cos_tuple, sin_tuple = create_rope_freqs_mlx(max_seq_len, head_dim)

        # Compare each component (looser tolerance for precision differences)
        np.testing.assert_allclose(
            np.array(cos_tuple[0]),
            torch_freqs_f.real.numpy(),
            rtol=1e-4,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            np.array(cos_tuple[1]),
            torch_freqs_h.real.numpy(),
            rtol=1e-4,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            np.array(cos_tuple[2]),
            torch_freqs_w.real.numpy(),
            rtol=1e-4,
            atol=1e-4,
        )


class TestRopeApplyMLX:
    """Tests for rope_apply_mlx comparing against PyTorch rope_apply."""

    def test_basic_shape(self):
        """Test that rope_apply produces correct output shape."""
        batch_size = 2
        seq_len = 256
        num_heads = 16
        head_dim = 128

        # Create input
        np_input = np.random.randn(
            batch_size, seq_len, num_heads, head_dim
        ).astype(np.float32)

        # Create grid sizes: (F, H, W) such that F*H*W <= seq_len
        # e.g., for video: 16 frames x 4 height x 4 width = 256
        grid_sizes = np.array([[16, 4, 4], [8, 4, 8]], dtype=np.int64)

        # Create frequencies
        cos_tuple, sin_tuple = create_rope_freqs_mlx(1024, head_dim)

        # Apply RoPE
        mlx_input = mx.array(np_input)
        mlx_grid = mx.array(grid_sizes)
        mlx_output = rope_apply_mlx(mlx_input, mlx_grid, cos_tuple, sin_tuple)
        mx.eval(mlx_output)

        # Check output shape
        assert mlx_output.shape == (batch_size, seq_len, num_heads, head_dim)

    def test_pytorch_equivalence_single_sample(self):
        """Test that MLX output matches PyTorch for a single sample."""
        batch_size = 1
        num_heads = 16
        head_dim = 128

        # Grid: 8 frames x 6 height x 8 width = 384 positions
        f, h, w = 8, 6, 8
        seq_len = f * h * w

        # Create input
        np.random.seed(42)
        np_input = np.random.randn(
            batch_size, seq_len, num_heads, head_dim
        ).astype(np.float32)

        grid_sizes_np = np.array([[f, h, w]], dtype=np.int64)

        # PyTorch reference
        torch_freqs = torch.cat([
            rope_params(1024, head_dim - 4 * (head_dim // 6)),
            rope_params(1024, 2 * (head_dim // 6)),
            rope_params(1024, 2 * (head_dim // 6))
        ], dim=1)
        torch_input = torch.from_numpy(np_input)
        torch_grid = torch.from_numpy(grid_sizes_np)
        with torch.no_grad():
            torch_output = rope_apply(torch_input, torch_grid, torch_freqs)
            torch_output_np = torch_output.numpy()

        # MLX implementation
        cos_tuple, sin_tuple = create_rope_freqs_mlx(1024, head_dim)
        mlx_input = mx.array(np_input)
        mlx_grid = mx.array(grid_sizes_np)
        mlx_output = rope_apply_mlx(mlx_input, mlx_grid, cos_tuple, sin_tuple)
        mx.eval(mlx_output)
        mlx_output_np = np.array(mlx_output)

        # Compare outputs - using 1e-4 tolerance as specified
        np.testing.assert_allclose(
            mlx_output_np,
            torch_output_np,
            rtol=1e-4,
            atol=1e-4,
            err_msg="MLX rope_apply output doesn't match PyTorch"
        )

    def test_pytorch_equivalence_batch(self):
        """Test that MLX output matches PyTorch for a batch of samples."""
        batch_size = 4
        num_heads = 16
        head_dim = 128
        max_seq_len = 512

        # Different grid sizes for each sample
        grid_sizes_list = [
            (8, 6, 8),   # 384
            (4, 8, 8),   # 256
            (16, 4, 4),  # 256
            (8, 8, 4),   # 256
        ]

        # Create padded input
        np.random.seed(123)
        np_input = np.random.randn(
            batch_size, max_seq_len, num_heads, head_dim
        ).astype(np.float32)

        grid_sizes_np = np.array(grid_sizes_list, dtype=np.int64)

        # PyTorch reference
        torch_freqs = torch.cat([
            rope_params(1024, head_dim - 4 * (head_dim // 6)),
            rope_params(1024, 2 * (head_dim // 6)),
            rope_params(1024, 2 * (head_dim // 6))
        ], dim=1)
        torch_input = torch.from_numpy(np_input)
        torch_grid = torch.from_numpy(grid_sizes_np)
        with torch.no_grad():
            torch_output = rope_apply(torch_input, torch_grid, torch_freqs)
            torch_output_np = torch_output.numpy()

        # MLX implementation
        cos_tuple, sin_tuple = create_rope_freqs_mlx(1024, head_dim)
        mlx_input = mx.array(np_input)
        mlx_grid = mx.array(grid_sizes_np)
        mlx_output = rope_apply_mlx(mlx_input, mlx_grid, cos_tuple, sin_tuple)
        mx.eval(mlx_output)
        mlx_output_np = np.array(mlx_output)

        # Compare outputs
        np.testing.assert_allclose(
            mlx_output_np,
            torch_output_np,
            rtol=1e-4,
            atol=1e-4,
            err_msg="MLX batch rope_apply output doesn't match PyTorch"
        )

    def test_variable_sequence_lengths(self):
        """Test with variable sequence lengths (padding in batch)."""
        batch_size = 2
        num_heads = 16
        head_dim = 128

        # First sample: 8*4*4=128, second sample: 4*4*4=64
        grid_sizes_list = [(8, 4, 4), (4, 4, 4)]
        max_seq_len = max(f * h * w for f, h, w in grid_sizes_list)  # 128

        np.random.seed(456)
        np_input = np.random.randn(
            batch_size, max_seq_len, num_heads, head_dim
        ).astype(np.float32)

        grid_sizes_np = np.array(grid_sizes_list, dtype=np.int64)

        # PyTorch reference
        torch_freqs = torch.cat([
            rope_params(1024, head_dim - 4 * (head_dim // 6)),
            rope_params(1024, 2 * (head_dim // 6)),
            rope_params(1024, 2 * (head_dim // 6))
        ], dim=1)
        torch_input = torch.from_numpy(np_input)
        torch_grid = torch.from_numpy(grid_sizes_np)
        with torch.no_grad():
            torch_output = rope_apply(torch_input, torch_grid, torch_freqs)
            torch_output_np = torch_output.numpy()

        # MLX implementation
        cos_tuple, sin_tuple = create_rope_freqs_mlx(1024, head_dim)
        mlx_input = mx.array(np_input)
        mlx_grid = mx.array(grid_sizes_np)
        mlx_output = rope_apply_mlx(mlx_input, mlx_grid, cos_tuple, sin_tuple)
        mx.eval(mlx_output)
        mlx_output_np = np.array(mlx_output)

        # Compare outputs
        np.testing.assert_allclose(
            mlx_output_np,
            torch_output_np,
            rtol=1e-4,
            atol=1e-4,
        )

    def test_padding_preserved(self):
        """Test that padding positions beyond seq_len are preserved."""
        batch_size = 1
        num_heads = 4
        head_dim = 128
        max_seq_len = 256

        # Grid: 4*4*4 = 64, so positions 64-255 are padding
        f, h, w = 4, 4, 4
        seq_len = f * h * w  # 64

        np.random.seed(789)
        np_input = np.random.randn(
            batch_size, max_seq_len, num_heads, head_dim
        ).astype(np.float32)

        grid_sizes_np = np.array([[f, h, w]], dtype=np.int64)

        # MLX implementation
        cos_tuple, sin_tuple = create_rope_freqs_mlx(1024, head_dim)
        mlx_input = mx.array(np_input)
        mlx_grid = mx.array(grid_sizes_np)
        mlx_output = rope_apply_mlx(mlx_input, mlx_grid, cos_tuple, sin_tuple)
        mx.eval(mlx_output)
        mlx_output_np = np.array(mlx_output)

        # Padding should be unchanged
        np.testing.assert_array_equal(
            mlx_output_np[0, seq_len:],
            np_input[0, seq_len:],
            err_msg="Padding positions were modified"
        )

    def test_rotation_property(self):
        """Test that rotation preserves magnitude (unit rotation property)."""
        batch_size = 1
        num_heads = 4
        head_dim = 128

        f, h, w = 4, 4, 4
        seq_len = f * h * w

        # Create unit vectors
        np_input = np.zeros((batch_size, seq_len, num_heads, head_dim), dtype=np.float32)
        # Set first pair to (1, 0) at first position
        np_input[0, 0, 0, 0] = 1.0

        grid_sizes_np = np.array([[f, h, w]], dtype=np.int64)

        # Apply RoPE
        cos_tuple, sin_tuple = create_rope_freqs_mlx(1024, head_dim)
        mlx_input = mx.array(np_input)
        mlx_grid = mx.array(grid_sizes_np)
        mlx_output = rope_apply_mlx(mlx_input, mlx_grid, cos_tuple, sin_tuple)
        mx.eval(mlx_output)
        mlx_output_np = np.array(mlx_output)

        # The rotated vector should have magnitude 1
        # Position 0 means rotation angle is 0, so output should be same as input
        # But let's check magnitude for a non-zero position
        np_input_pos10 = np.zeros((batch_size, seq_len, num_heads, head_dim), dtype=np.float32)
        np_input_pos10[0, 10, 0, 0] = 1.0
        np_input_pos10[0, 10, 0, 1] = 0.0

        mlx_input_pos10 = mx.array(np_input_pos10)
        mlx_output_pos10 = rope_apply_mlx(mlx_input_pos10, mlx_grid, cos_tuple, sin_tuple)
        mx.eval(mlx_output_pos10)
        out_np = np.array(mlx_output_pos10)

        # Check magnitude of first pair is preserved
        magnitude = np.sqrt(out_np[0, 10, 0, 0]**2 + out_np[0, 10, 0, 1]**2)
        np.testing.assert_allclose(magnitude, 1.0, rtol=1e-5)

    def test_dtype_float32_output(self):
        """Test that output is always float32."""
        batch_size = 1
        num_heads = 4
        head_dim = 128

        f, h, w = 4, 4, 4
        seq_len = f * h * w

        np_input = np.random.randn(
            batch_size, seq_len, num_heads, head_dim
        ).astype(np.float32)

        grid_sizes_np = np.array([[f, h, w]], dtype=np.int64)

        cos_tuple, sin_tuple = create_rope_freqs_mlx(1024, head_dim)
        mlx_input = mx.array(np_input)
        mlx_grid = mx.array(grid_sizes_np)
        mlx_output = rope_apply_mlx(mlx_input, mlx_grid, cos_tuple, sin_tuple)
        mx.eval(mlx_output)

        assert mlx_output.dtype == mx.float32


class TestRopeEndToEnd:
    """End-to-end tests simulating WanModel usage patterns."""

    def test_wanmodel_typical_shapes(self):
        """Test with typical shapes from WanModel inference."""
        # WanModel config: num_heads=16, dim=2048, head_dim=128
        batch_size = 1
        num_heads = 16
        head_dim = 128

        # 480p video: typical grid for 41 frames would be something like
        # (41, 30, 52) but that's large; use smaller for test
        f, h, w = 11, 15, 26  # 4290 positions
        seq_len = f * h * w

        np.random.seed(2024)
        np_input = np.random.randn(
            batch_size, seq_len, num_heads, head_dim
        ).astype(np.float32)

        grid_sizes_np = np.array([[f, h, w]], dtype=np.int64)

        # PyTorch reference (how WanModel does it)
        torch_freqs = torch.cat([
            rope_params(1024, head_dim - 4 * (head_dim // 6)),
            rope_params(1024, 2 * (head_dim // 6)),
            rope_params(1024, 2 * (head_dim // 6))
        ], dim=1)
        torch_input = torch.from_numpy(np_input)
        torch_grid = torch.from_numpy(grid_sizes_np)
        with torch.no_grad():
            torch_output = rope_apply(torch_input, torch_grid, torch_freqs)
            torch_output_np = torch_output.numpy()

        # MLX implementation
        cos_tuple, sin_tuple = create_rope_freqs_mlx(1024, head_dim)
        mlx_input = mx.array(np_input)
        mlx_grid = mx.array(grid_sizes_np)
        mlx_output = rope_apply_mlx(mlx_input, mlx_grid, cos_tuple, sin_tuple)
        mx.eval(mlx_output)
        mlx_output_np = np.array(mlx_output)

        # Compare
        np.testing.assert_allclose(
            mlx_output_np,
            torch_output_np,
            rtol=1e-4,
            atol=1e-4,
            err_msg="MLX doesn't match PyTorch for typical WanModel shapes"
        )

    def test_max_position_1024(self):
        """Test with positions up to max_seq_len=1024."""
        batch_size = 1
        num_heads = 4
        head_dim = 128

        # Use dimensions that multiply to close to 1024
        f, h, w = 8, 8, 16  # 1024 positions
        seq_len = f * h * w

        np.random.seed(1000)
        np_input = np.random.randn(
            batch_size, seq_len, num_heads, head_dim
        ).astype(np.float32)

        grid_sizes_np = np.array([[f, h, w]], dtype=np.int64)

        # PyTorch reference
        torch_freqs = torch.cat([
            rope_params(1024, head_dim - 4 * (head_dim // 6)),
            rope_params(1024, 2 * (head_dim // 6)),
            rope_params(1024, 2 * (head_dim // 6))
        ], dim=1)
        torch_input = torch.from_numpy(np_input)
        torch_grid = torch.from_numpy(grid_sizes_np)
        with torch.no_grad():
            torch_output = rope_apply(torch_input, torch_grid, torch_freqs)
            torch_output_np = torch_output.numpy()

        # MLX implementation
        cos_tuple, sin_tuple = create_rope_freqs_mlx(1024, head_dim)
        mlx_input = mx.array(np_input)
        mlx_grid = mx.array(grid_sizes_np)
        mlx_output = rope_apply_mlx(mlx_input, mlx_grid, cos_tuple, sin_tuple)
        mx.eval(mlx_output)
        mlx_output_np = np.array(mlx_output)

        np.testing.assert_allclose(
            mlx_output_np,
            torch_output_np,
            rtol=1e-4,
            atol=1e-4,
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
