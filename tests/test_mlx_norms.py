"""Unit tests for MLX normalization layers.

Tests that MLX implementations match PyTorch reference within tolerance.
"""

import numpy as np
import pytest

# Import frameworks
import mlx.core as mx
import torch

# Import implementations to test
from wan.backends.mlx.norms import WanRMSNormMLX, WanLayerNormMLX
from wan.modules.model import WanRMSNorm, WanLayerNorm


class TestWanRMSNormMLX:
    """Tests for WanRMSNormMLX comparing against PyTorch WanRMSNorm."""

    @pytest.fixture
    def dims(self):
        """Common dimension sizes to test."""
        return [128, 256, 512, 2048]

    @pytest.fixture
    def batch_sizes(self):
        """Common batch sizes to test."""
        return [1, 4, 8]

    @pytest.fixture
    def seq_lengths(self):
        """Common sequence lengths to test."""
        return [16, 64, 256]

    def test_basic_normalization(self):
        """Test that MLX RMSNorm produces correct output shape."""
        dim = 128
        batch_size = 2
        seq_len = 16

        # Create random input
        np_input = np.random.randn(batch_size, seq_len, dim).astype(np.float32)

        # MLX forward
        mlx_norm = WanRMSNormMLX(dim, eps=1e-5)
        mlx_input = mx.array(np_input)
        mlx_output = mlx_norm(mlx_input)
        mx.eval(mlx_output)

        # Check output shape
        assert mlx_output.shape == (batch_size, seq_len, dim)

    def test_pytorch_equivalence(self):
        """Test that MLX output matches PyTorch within 1e-5 tolerance."""
        dim = 2048  # Match WanModel hidden dim
        batch_size = 2
        seq_len = 64

        # Create random input
        np_input = np.random.randn(batch_size, seq_len, dim).astype(np.float32)

        # PyTorch reference
        torch_norm = WanRMSNorm(dim, eps=1e-5)
        torch_input = torch.from_numpy(np_input)
        with torch.no_grad():
            torch_output = torch_norm(torch_input).numpy()

        # MLX implementation
        mlx_norm = WanRMSNormMLX(dim, eps=1e-5)
        mlx_input = mx.array(np_input)
        mlx_output = mlx_norm(mlx_input)
        mx.eval(mlx_output)
        mlx_output_np = np.array(mlx_output)

        # Compare outputs
        np.testing.assert_allclose(
            mlx_output_np,
            torch_output,
            rtol=1e-5,
            atol=1e-5,
            err_msg="MLX output does not match PyTorch within 1e-5 tolerance"
        )

    def test_learnable_weight(self):
        """Test that learnable weight parameter has correct shape."""
        dim = 256

        mlx_norm = WanRMSNormMLX(dim, eps=1e-5)

        # Check weight shape
        assert mlx_norm.weight.shape == (dim,)

        # Check weight is initialized to ones
        np.testing.assert_allclose(
            np.array(mlx_norm.weight),
            np.ones(dim),
            rtol=1e-7
        )

    def test_custom_weight(self):
        """Test that custom weights are applied correctly."""
        dim = 128
        batch_size = 2
        seq_len = 16

        np_input = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
        np_weight = np.random.randn(dim).astype(np.float32)

        # PyTorch with custom weight
        torch_norm = WanRMSNorm(dim, eps=1e-5)
        torch_norm.weight.data = torch.from_numpy(np_weight)
        torch_input = torch.from_numpy(np_input)
        with torch.no_grad():
            torch_output = torch_norm(torch_input).numpy()

        # MLX with custom weight
        mlx_norm = WanRMSNormMLX(dim, eps=1e-5)
        mlx_norm.weight = mx.array(np_weight)
        mlx_input = mx.array(np_input)
        mlx_output = mlx_norm(mlx_input)
        mx.eval(mlx_output)
        mlx_output_np = np.array(mlx_output)

        # Compare outputs
        np.testing.assert_allclose(
            mlx_output_np,
            torch_output,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Custom weight application differs between MLX and PyTorch"
        )

    def test_eps_parameter(self):
        """Test that eps=1e-5 is correctly used."""
        dim = 128

        mlx_norm = WanRMSNormMLX(dim, eps=1e-5)
        assert mlx_norm.eps == 1e-5

        # Test with different eps
        mlx_norm_custom = WanRMSNormMLX(dim, eps=1e-6)
        assert mlx_norm_custom.eps == 1e-6

    def test_various_dimensions(self, dims, batch_sizes, seq_lengths):
        """Test correctness across various input shapes."""
        for dim in dims:
            for batch_size in batch_sizes:
                for seq_len in seq_lengths:
                    np_input = np.random.randn(
                        batch_size, seq_len, dim
                    ).astype(np.float32)

                    # PyTorch reference
                    torch_norm = WanRMSNorm(dim, eps=1e-5)
                    torch_input = torch.from_numpy(np_input)
                    with torch.no_grad():
                        torch_output = torch_norm(torch_input).numpy()

                    # MLX implementation
                    mlx_norm = WanRMSNormMLX(dim, eps=1e-5)
                    mlx_input = mx.array(np_input)
                    mlx_output = mlx_norm(mlx_input)
                    mx.eval(mlx_output)
                    mlx_output_np = np.array(mlx_output)

                    # Compare outputs
                    np.testing.assert_allclose(
                        mlx_output_np,
                        torch_output,
                        rtol=1e-5,
                        atol=1e-5,
                        err_msg=f"Mismatch for shape ({batch_size}, {seq_len}, {dim})"
                    )

    def test_numerical_stability_near_zero(self):
        """Test numerical stability with near-zero inputs."""
        dim = 128
        batch_size = 2
        seq_len = 16

        # Create input with very small values
        np_input = np.random.randn(batch_size, seq_len, dim).astype(np.float32) * 1e-10

        # PyTorch reference
        torch_norm = WanRMSNorm(dim, eps=1e-5)
        torch_input = torch.from_numpy(np_input)
        with torch.no_grad():
            torch_output = torch_norm(torch_input).numpy()

        # MLX implementation
        mlx_norm = WanRMSNormMLX(dim, eps=1e-5)
        mlx_input = mx.array(np_input)
        mlx_output = mlx_norm(mlx_input)
        mx.eval(mlx_output)
        mlx_output_np = np.array(mlx_output)

        # Check no NaN/Inf values
        assert not np.any(np.isnan(mlx_output_np)), "MLX output contains NaN"
        assert not np.any(np.isinf(mlx_output_np)), "MLX output contains Inf"

        # Compare outputs (with looser tolerance for edge cases)
        np.testing.assert_allclose(
            mlx_output_np,
            torch_output,
            rtol=1e-4,
            atol=1e-4
        )

    def test_dtype_preservation(self):
        """Test that output dtype matches input dtype."""
        dim = 128
        batch_size = 2
        seq_len = 16

        mlx_norm = WanRMSNormMLX(dim, eps=1e-5)

        # Test float32
        np_input_f32 = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
        mlx_input_f32 = mx.array(np_input_f32)
        mlx_output_f32 = mlx_norm(mlx_input_f32)
        mx.eval(mlx_output_f32)
        assert mlx_output_f32.dtype == mx.float32

        # Test float16
        mlx_input_f16 = mlx_input_f32.astype(mx.float16)
        mlx_output_f16 = mlx_norm(mlx_input_f16)
        mx.eval(mlx_output_f16)
        assert mlx_output_f16.dtype == mx.float16


class TestWanLayerNormMLX:
    """Tests for WanLayerNormMLX comparing against PyTorch WanLayerNorm."""

    @pytest.fixture
    def dims(self):
        """Common dimension sizes to test."""
        return [128, 256, 512, 2048]

    @pytest.fixture
    def batch_sizes(self):
        """Common batch sizes to test."""
        return [1, 4, 8]

    @pytest.fixture
    def seq_lengths(self):
        """Common sequence lengths to test."""
        return [16, 64, 256]

    def test_basic_normalization(self):
        """Test that MLX LayerNorm produces correct output shape."""
        dim = 128
        batch_size = 2
        seq_len = 16

        # Create random input
        np_input = np.random.randn(batch_size, seq_len, dim).astype(np.float32)

        # MLX forward
        mlx_norm = WanLayerNormMLX(dim, eps=1e-6)
        mlx_input = mx.array(np_input)
        mlx_output = mlx_norm(mlx_input)
        mx.eval(mlx_output)

        # Check output shape
        assert mlx_output.shape == (batch_size, seq_len, dim)

    def test_pytorch_equivalence_no_affine(self):
        """Test that MLX output matches PyTorch within 1e-5 tolerance (no affine)."""
        dim = 2048  # Match WanModel hidden dim
        batch_size = 2
        seq_len = 64

        # Create random input
        np_input = np.random.randn(batch_size, seq_len, dim).astype(np.float32)

        # PyTorch reference (elementwise_affine=False is default)
        torch_norm = WanLayerNorm(dim, eps=1e-6, elementwise_affine=False)
        torch_input = torch.from_numpy(np_input)
        with torch.no_grad():
            torch_output = torch_norm(torch_input).numpy()

        # MLX implementation
        mlx_norm = WanLayerNormMLX(dim, eps=1e-6, elementwise_affine=False)
        mlx_input = mx.array(np_input)
        mlx_output = mlx_norm(mlx_input)
        mx.eval(mlx_output)
        mlx_output_np = np.array(mlx_output)

        # Compare outputs
        np.testing.assert_allclose(
            mlx_output_np,
            torch_output,
            rtol=1e-5,
            atol=1e-5,
            err_msg="MLX output does not match PyTorch within 1e-5 tolerance"
        )

    def test_pytorch_equivalence_with_affine(self):
        """Test that MLX output matches PyTorch with elementwise_affine=True."""
        dim = 512
        batch_size = 2
        seq_len = 32

        # Create random input
        np_input = np.random.randn(batch_size, seq_len, dim).astype(np.float32)

        # PyTorch reference with affine
        torch_norm = WanLayerNorm(dim, eps=1e-6, elementwise_affine=True)
        torch_input = torch.from_numpy(np_input)
        with torch.no_grad():
            torch_output = torch_norm(torch_input).numpy()

        # MLX implementation with affine
        mlx_norm = WanLayerNormMLX(dim, eps=1e-6, elementwise_affine=True)
        mlx_input = mx.array(np_input)
        mlx_output = mlx_norm(mlx_input)
        mx.eval(mlx_output)
        mlx_output_np = np.array(mlx_output)

        # Compare outputs (initialized weights/biases should match)
        np.testing.assert_allclose(
            mlx_output_np,
            torch_output,
            rtol=1e-5,
            atol=1e-5,
            err_msg="MLX output with affine does not match PyTorch"
        )

    def test_custom_affine_weights(self):
        """Test that custom weight/bias are applied correctly."""
        dim = 128
        batch_size = 2
        seq_len = 16

        np_input = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
        np_weight = np.random.randn(dim).astype(np.float32)
        np_bias = np.random.randn(dim).astype(np.float32)

        # PyTorch with custom weight/bias
        torch_norm = WanLayerNorm(dim, eps=1e-6, elementwise_affine=True)
        torch_norm.weight.data = torch.from_numpy(np_weight)
        torch_norm.bias.data = torch.from_numpy(np_bias)
        torch_input = torch.from_numpy(np_input)
        with torch.no_grad():
            torch_output = torch_norm(torch_input).numpy()

        # MLX with custom weight/bias
        mlx_norm = WanLayerNormMLX(dim, eps=1e-6, elementwise_affine=True)
        mlx_norm.weight = mx.array(np_weight)
        mlx_norm.bias = mx.array(np_bias)
        mlx_input = mx.array(np_input)
        mlx_output = mlx_norm(mlx_input)
        mx.eval(mlx_output)
        mlx_output_np = np.array(mlx_output)

        # Compare outputs
        np.testing.assert_allclose(
            mlx_output_np,
            torch_output,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Custom weight/bias application differs between MLX and PyTorch"
        )

    def test_eps_default(self):
        """Test that eps=1e-6 is the default (matching WanLayerNorm)."""
        dim = 128

        mlx_norm = WanLayerNormMLX(dim)
        assert mlx_norm.eps == 1e-6

        # Test with custom eps
        mlx_norm_custom = WanLayerNormMLX(dim, eps=1e-5)
        assert mlx_norm_custom.eps == 1e-5

    def test_elementwise_affine_default(self):
        """Test that elementwise_affine=False is the default (matching WanLayerNorm)."""
        dim = 128

        mlx_norm = WanLayerNormMLX(dim)
        assert mlx_norm.elementwise_affine is False
        assert mlx_norm.weight is None
        assert mlx_norm.bias is None

        # With affine enabled
        mlx_norm_affine = WanLayerNormMLX(dim, elementwise_affine=True)
        assert mlx_norm_affine.elementwise_affine is True
        assert mlx_norm_affine.weight is not None
        assert mlx_norm_affine.bias is not None

    def test_affine_weight_init(self):
        """Test that affine weight is initialized to ones, bias to zeros."""
        dim = 256

        mlx_norm = WanLayerNormMLX(dim, elementwise_affine=True)

        # Check weight is initialized to ones
        np.testing.assert_allclose(
            np.array(mlx_norm.weight),
            np.ones(dim),
            rtol=1e-7
        )

        # Check bias is initialized to zeros
        np.testing.assert_allclose(
            np.array(mlx_norm.bias),
            np.zeros(dim),
            rtol=1e-7
        )

    def test_various_dimensions(self, dims, batch_sizes, seq_lengths):
        """Test correctness across various input shapes."""
        for dim in dims:
            for batch_size in batch_sizes:
                for seq_len in seq_lengths:
                    np_input = np.random.randn(
                        batch_size, seq_len, dim
                    ).astype(np.float32)

                    # PyTorch reference
                    torch_norm = WanLayerNorm(dim, eps=1e-6, elementwise_affine=False)
                    torch_input = torch.from_numpy(np_input)
                    with torch.no_grad():
                        torch_output = torch_norm(torch_input).numpy()

                    # MLX implementation
                    mlx_norm = WanLayerNormMLX(dim, eps=1e-6, elementwise_affine=False)
                    mlx_input = mx.array(np_input)
                    mlx_output = mlx_norm(mlx_input)
                    mx.eval(mlx_output)
                    mlx_output_np = np.array(mlx_output)

                    # Compare outputs
                    np.testing.assert_allclose(
                        mlx_output_np,
                        torch_output,
                        rtol=1e-5,
                        atol=1e-5,
                        err_msg=f"Mismatch for shape ({batch_size}, {seq_len}, {dim})"
                    )

    def test_numerical_stability_near_zero(self):
        """Test numerical stability with near-zero inputs."""
        dim = 128
        batch_size = 2
        seq_len = 16

        # Create input with very small values
        np_input = np.random.randn(batch_size, seq_len, dim).astype(np.float32) * 1e-10

        # PyTorch reference
        torch_norm = WanLayerNorm(dim, eps=1e-6, elementwise_affine=False)
        torch_input = torch.from_numpy(np_input)
        with torch.no_grad():
            torch_output = torch_norm(torch_input).numpy()

        # MLX implementation
        mlx_norm = WanLayerNormMLX(dim, eps=1e-6, elementwise_affine=False)
        mlx_input = mx.array(np_input)
        mlx_output = mlx_norm(mlx_input)
        mx.eval(mlx_output)
        mlx_output_np = np.array(mlx_output)

        # Check no NaN/Inf values
        assert not np.any(np.isnan(mlx_output_np)), "MLX output contains NaN"
        assert not np.any(np.isinf(mlx_output_np)), "MLX output contains Inf"

        # Compare outputs (with looser tolerance for edge cases)
        np.testing.assert_allclose(
            mlx_output_np,
            torch_output,
            rtol=1e-4,
            atol=1e-4
        )

    def test_dtype_preservation(self):
        """Test that output dtype matches input dtype."""
        dim = 128
        batch_size = 2
        seq_len = 16

        mlx_norm = WanLayerNormMLX(dim, eps=1e-6)

        # Test float32
        np_input_f32 = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
        mlx_input_f32 = mx.array(np_input_f32)
        mlx_output_f32 = mlx_norm(mlx_input_f32)
        mx.eval(mlx_output_f32)
        assert mlx_output_f32.dtype == mx.float32

        # Test float16
        mlx_input_f16 = mlx_input_f32.astype(mx.float16)
        mlx_output_f16 = mlx_norm(mlx_input_f16)
        mx.eval(mlx_output_f16)
        assert mlx_output_f16.dtype == mx.float16


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
