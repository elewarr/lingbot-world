"""Unit tests for MLX HeadMLX layer.

Tests that MLX implementation matches PyTorch reference within tolerance.

Key test areas:
- Initialization and shapes
- Output dimension calculation from patch_size
- 2-way modulation mechanism
- Full head PyTorch equivalence
"""

import math

import numpy as np
import pytest

import mlx.core as mx
import mlx.nn as nn_mlx
import torch
import torch.nn as nn

from wan.backends.mlx.head import HeadMLX
from wan.modules.model import Head


def copy_weights_head(mlx_head: HeadMLX, torch_head: Head) -> None:
    """Copy weights from PyTorch Head to MLX HeadMLX.

    Args:
        mlx_head: Target MLX head module
        torch_head: Source PyTorch head module
    """
    # Copy head linear layer weights
    mlx_head.head.weight = mx.array(torch_head.head.weight.detach().numpy())
    mlx_head.head.bias = mx.array(torch_head.head.bias.detach().numpy())

    # Copy modulation parameter
    mlx_head.modulation = mx.array(torch_head.modulation.detach().numpy())

    # norm has no learnable params when elementwise_affine=False


class TestHeadMLXInitialization:
    """Tests for HeadMLX initialization and structure."""

    @pytest.fixture
    def default_config(self):
        """Default configuration matching WanModel."""
        return {
            'dim': 2048,
            'out_dim': 16,
            'patch_size': (1, 2, 2),
            'eps': 1e-6,
        }

    @pytest.fixture
    def small_config(self):
        """Smaller configuration for faster tests."""
        return {
            'dim': 256,
            'out_dim': 16,
            'patch_size': (1, 2, 2),
            'eps': 1e-6,
        }

    def test_initialization_default(self, default_config):
        """Test that HeadMLX initializes with default config."""
        head = HeadMLX(**default_config)

        assert head.dim == default_config['dim']
        assert head.out_dim == default_config['out_dim']
        assert head.patch_size == default_config['patch_size']
        assert head.eps == default_config['eps']

    def test_initialization_small(self, small_config):
        """Test that HeadMLX initializes with small config."""
        head = HeadMLX(**small_config)

        assert head.dim == small_config['dim']
        assert head.out_dim == small_config['out_dim']
        assert head.patch_size == small_config['patch_size']

    def test_output_dim_calculation(self, small_config):
        """Test that output dimension is correctly calculated from patch_size."""
        head = HeadMLX(**small_config)
        dim = small_config['dim']
        out_dim = small_config['out_dim']
        patch_size = small_config['patch_size']

        # Expected: prod(patch_size) * out_dim = 1*2*2*16 = 64
        expected_full_out_dim = math.prod(patch_size) * out_dim

        # Check linear layer output dimension
        # MLX Linear weight shape: [out_features, in_features]
        assert head.head.weight.shape == (expected_full_out_dim, dim)
        assert head.head.bias.shape == (expected_full_out_dim,)

    def test_modulation_shape(self, small_config):
        """Test that modulation parameter has correct shape."""
        head = HeadMLX(**small_config)
        dim = small_config['dim']

        # Modulation should be (1, 2, dim) - 2 for shift and scale
        assert head.modulation.shape == (1, 2, dim)

    def test_different_patch_sizes(self):
        """Test with different patch_size configurations."""
        dim = 256
        out_dim = 16

        # Test various patch sizes
        test_cases = [
            (1, 2, 2),   # Default: 1*2*2*16 = 64
            (1, 1, 1),   # Minimal: 1*1*1*16 = 16
            (2, 2, 2),   # Larger: 2*2*2*16 = 128
            (1, 4, 4),   # Spatial-focused: 1*4*4*16 = 256
        ]

        for patch_size in test_cases:
            head = HeadMLX(dim=dim, out_dim=out_dim, patch_size=patch_size)
            expected_out = math.prod(patch_size) * out_dim

            assert head.head.weight.shape == (expected_out, dim), \
                f"Failed for patch_size={patch_size}"


class TestHeadMLXForward:
    """Tests for HeadMLX forward pass."""

    @pytest.fixture
    def small_config(self):
        return {
            'dim': 256,
            'out_dim': 16,
            'patch_size': (1, 2, 2),
            'eps': 1e-6,
        }

    def test_output_shape(self, small_config):
        """Test that forward pass produces correct output shape."""
        head = HeadMLX(**small_config)
        dim = small_config['dim']
        out_dim = small_config['out_dim']
        patch_size = small_config['patch_size']

        batch_size = 2
        seq_len = 64
        expected_full_out_dim = math.prod(patch_size) * out_dim  # 64

        np.random.seed(42)
        np_x = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
        np_e = np.random.randn(batch_size, seq_len, dim).astype(np.float32)

        mlx_x = mx.array(np_x)
        mlx_e = mx.array(np_e)

        output = head(mlx_x, mlx_e)
        mx.eval(output)

        assert output.shape == (batch_size, seq_len, expected_full_out_dim)

    def test_no_nan_inf(self, small_config):
        """Test that forward pass doesn't produce NaN or Inf."""
        head = HeadMLX(**small_config)
        dim = small_config['dim']

        batch_size = 2
        seq_len = 64

        np.random.seed(42)
        np_x = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
        np_e = np.random.randn(batch_size, seq_len, dim).astype(np.float32)

        mlx_x = mx.array(np_x)
        mlx_e = mx.array(np_e)

        output = head(mlx_x, mlx_e)
        mx.eval(output)
        output_np = np.array(output)

        assert not np.any(np.isnan(output_np)), "Output contains NaN"
        assert not np.any(np.isinf(output_np)), "Output contains Inf"


class TestHeadMLXPyTorchEquivalence:
    """Tests comparing MLX HeadMLX to PyTorch reference."""

    @pytest.fixture
    def small_config(self):
        return {
            'dim': 256,
            'out_dim': 16,
            'patch_size': (1, 2, 2),
            'eps': 1e-6,
        }

    def test_pytorch_equivalence_basic(self, small_config):
        """Test that MLX output matches PyTorch for basic inputs."""
        batch_size = 1
        dim = small_config['dim']
        out_dim = small_config['out_dim']
        patch_size = small_config['patch_size']
        seq_len = 64

        # Create shared inputs
        np.random.seed(123)
        np_x = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
        np_e = np.random.randn(batch_size, seq_len, dim).astype(np.float32)

        # ---- PyTorch Reference ----
        torch_head = Head(**small_config)

        torch_x = torch.from_numpy(np_x)
        torch_e = torch.from_numpy(np_e)

        with torch.no_grad():
            torch_output = torch_head(torch_x, torch_e)
        torch_output_np = torch_output.numpy()

        # ---- MLX Implementation ----
        mlx_head = HeadMLX(**small_config)
        copy_weights_head(mlx_head, torch_head)

        mlx_x = mx.array(np_x)
        mlx_e = mx.array(np_e)

        mlx_output = mlx_head(mlx_x, mlx_e)
        mx.eval(mlx_output)
        mlx_output_np = np.array(mlx_output)

        # Compare outputs
        np.testing.assert_allclose(
            mlx_output_np,
            torch_output_np,
            rtol=1e-4,
            atol=1e-4,
            err_msg="MLX head output doesn't match PyTorch"
        )

    def test_pytorch_equivalence_batch(self, small_config):
        """Test equivalence with batched inputs."""
        batch_size = 4
        dim = small_config['dim']
        seq_len = 64

        np.random.seed(456)
        np_x = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
        np_e = np.random.randn(batch_size, seq_len, dim).astype(np.float32)

        # ---- PyTorch Reference ----
        torch_head = Head(**small_config)

        torch_x = torch.from_numpy(np_x)
        torch_e = torch.from_numpy(np_e)

        with torch.no_grad():
            torch_output = torch_head(torch_x, torch_e)
        torch_output_np = torch_output.numpy()

        # ---- MLX Implementation ----
        mlx_head = HeadMLX(**small_config)
        copy_weights_head(mlx_head, torch_head)

        mlx_x = mx.array(np_x)
        mlx_e = mx.array(np_e)

        mlx_output = mlx_head(mlx_x, mlx_e)
        mx.eval(mlx_output)
        mlx_output_np = np.array(mlx_output)

        np.testing.assert_allclose(
            mlx_output_np,
            torch_output_np,
            rtol=1e-4,
            atol=1e-4,
            err_msg="MLX head batch output doesn't match PyTorch"
        )

    def test_pytorch_equivalence_wanmodel_config(self):
        """Test with exact WanModel configuration (dim=2048, out_dim=16)."""
        config = {
            'dim': 2048,
            'out_dim': 16,
            'patch_size': (1, 2, 2),
            'eps': 1e-6,
        }
        batch_size = 1
        dim = config['dim']
        seq_len = 64  # Smaller for faster test

        np.random.seed(789)
        np_x = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
        np_e = np.random.randn(batch_size, seq_len, dim).astype(np.float32)

        # ---- PyTorch Reference ----
        torch_head = Head(**config)

        torch_x = torch.from_numpy(np_x)
        torch_e = torch.from_numpy(np_e)

        with torch.no_grad():
            torch_output = torch_head(torch_x, torch_e)
        torch_output_np = torch_output.numpy()

        # ---- MLX Implementation ----
        mlx_head = HeadMLX(**config)
        copy_weights_head(mlx_head, torch_head)

        mlx_x = mx.array(np_x)
        mlx_e = mx.array(np_e)

        mlx_output = mlx_head(mlx_x, mlx_e)
        mx.eval(mlx_output)
        mlx_output_np = np.array(mlx_output)

        np.testing.assert_allclose(
            mlx_output_np,
            torch_output_np,
            rtol=1e-4,
            atol=1e-4,
            err_msg="MLX head doesn't match PyTorch for WanModel config"
        )

    def test_pytorch_equivalence_various_patch_sizes(self, small_config):
        """Test equivalence with different patch_size configurations."""
        dim = small_config['dim']
        out_dim = small_config['out_dim']
        batch_size = 1
        seq_len = 32

        patch_sizes = [
            (1, 2, 2),  # Default
            (1, 1, 1),  # Minimal
            (2, 2, 2),  # Larger
        ]

        for patch_size in patch_sizes:
            config = {
                'dim': dim,
                'out_dim': out_dim,
                'patch_size': patch_size,
                'eps': 1e-6,
            }

            np.random.seed(111)
            np_x = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
            np_e = np.random.randn(batch_size, seq_len, dim).astype(np.float32)

            # PyTorch
            torch_head = Head(**config)
            torch_x = torch.from_numpy(np_x)
            torch_e = torch.from_numpy(np_e)

            with torch.no_grad():
                torch_output = torch_head(torch_x, torch_e)
            torch_output_np = torch_output.numpy()

            # MLX
            mlx_head = HeadMLX(**config)
            copy_weights_head(mlx_head, torch_head)

            mlx_x = mx.array(np_x)
            mlx_e = mx.array(np_e)

            mlx_output = mlx_head(mlx_x, mlx_e)
            mx.eval(mlx_output)
            mlx_output_np = np.array(mlx_output)

            np.testing.assert_allclose(
                mlx_output_np,
                torch_output_np,
                rtol=1e-4,
                atol=1e-4,
                err_msg=f"MLX head doesn't match PyTorch for patch_size={patch_size}"
            )


class TestHeadMLXModulation:
    """Tests for 2-way modulation mechanism."""

    @pytest.fixture
    def small_config(self):
        return {
            'dim': 256,
            'out_dim': 16,
            'patch_size': (1, 2, 2),
            'eps': 1e-6,
        }

    def test_modulation_affects_output(self, small_config):
        """Test that modulation actually affects the output."""
        head = HeadMLX(**small_config)
        dim = small_config['dim']

        batch_size = 1
        seq_len = 16

        np.random.seed(42)
        np_x = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
        np_e1 = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
        np_e2 = np.random.randn(batch_size, seq_len, dim).astype(np.float32) * 10

        mlx_x = mx.array(np_x)
        mlx_e1 = mx.array(np_e1)
        mlx_e2 = mx.array(np_e2)

        output1 = head(mlx_x, mlx_e1)
        output2 = head(mlx_x, mlx_e2)
        mx.eval(output1, output2)

        # Different time embeddings should produce different outputs
        assert not np.allclose(np.array(output1), np.array(output2)), \
            "Different time embeddings should produce different outputs"

    def test_zero_modulation_effect(self, small_config):
        """Test behavior with zero time embedding."""
        head = HeadMLX(**small_config)
        dim = small_config['dim']

        batch_size = 1
        seq_len = 16

        np.random.seed(42)
        np_x = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
        np_e_zero = np.zeros((batch_size, seq_len, dim), dtype=np.float32)

        mlx_x = mx.array(np_x)
        mlx_e_zero = mx.array(np_e_zero)

        output = head(mlx_x, mlx_e_zero)
        mx.eval(output)
        output_np = np.array(output)

        # Output should still be valid (no NaN/Inf)
        assert not np.any(np.isnan(output_np)), "Zero embedding produced NaN"
        assert not np.any(np.isinf(output_np)), "Zero embedding produced Inf"


class TestHeadMLXNumericalStability:
    """Tests for numerical stability."""

    @pytest.fixture
    def small_config(self):
        return {
            'dim': 256,
            'out_dim': 16,
            'patch_size': (1, 2, 2),
            'eps': 1e-6,
        }

    def test_small_values(self, small_config):
        """Test numerical stability with small input values."""
        head = HeadMLX(**small_config)
        dim = small_config['dim']

        batch_size = 1
        seq_len = 32

        np.random.seed(222)
        np_x = np.random.randn(batch_size, seq_len, dim).astype(np.float32) * 1e-4
        np_e = np.random.randn(batch_size, seq_len, dim).astype(np.float32) * 1e-4

        mlx_x = mx.array(np_x)
        mlx_e = mx.array(np_e)

        output = head(mlx_x, mlx_e)
        mx.eval(output)
        output_np = np.array(output)

        assert not np.any(np.isnan(output_np)), "Small input produced NaN"
        assert not np.any(np.isinf(output_np)), "Small input produced Inf"

    def test_large_values(self, small_config):
        """Test numerical stability with large input values."""
        head = HeadMLX(**small_config)
        dim = small_config['dim']

        batch_size = 1
        seq_len = 32

        np.random.seed(333)
        np_x = np.random.randn(batch_size, seq_len, dim).astype(np.float32) * 10
        np_e = np.random.randn(batch_size, seq_len, dim).astype(np.float32) * 10

        mlx_x = mx.array(np_x)
        mlx_e = mx.array(np_e)

        output = head(mlx_x, mlx_e)
        mx.eval(output)
        output_np = np.array(output)

        assert not np.any(np.isnan(output_np)), "Large input produced NaN"
        assert not np.any(np.isinf(output_np)), "Large input produced Inf"

    def test_deterministic_output(self, small_config):
        """Test that the same input produces the same output."""
        head = HeadMLX(**small_config)
        dim = small_config['dim']

        batch_size = 1
        seq_len = 32

        np.random.seed(444)
        np_x = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
        np_e = np.random.randn(batch_size, seq_len, dim).astype(np.float32)

        mlx_x = mx.array(np_x)
        mlx_e = mx.array(np_e)

        output1 = head(mlx_x, mlx_e)
        mx.eval(output1)

        output2 = head(mlx_x, mlx_e)
        mx.eval(output2)

        np.testing.assert_array_equal(
            np.array(output1),
            np.array(output2),
            err_msg="Same input produced different outputs"
        )


class TestHeadMLXBatching:
    """Tests for batch processing."""

    @pytest.fixture
    def small_config(self):
        return {
            'dim': 256,
            'out_dim': 16,
            'patch_size': (1, 2, 2),
            'eps': 1e-6,
        }

    def test_batch_processing(self, small_config):
        """Test that batched inputs are processed correctly."""
        head = HeadMLX(**small_config)
        dim = small_config['dim']
        patch_size = small_config['patch_size']
        out_dim = small_config['out_dim']

        batch_size = 4
        seq_len = 64
        expected_out_dim = math.prod(patch_size) * out_dim

        np.random.seed(555)
        np_x = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
        np_e = np.random.randn(batch_size, seq_len, dim).astype(np.float32)

        mlx_x = mx.array(np_x)
        mlx_e = mx.array(np_e)

        output = head(mlx_x, mlx_e)
        mx.eval(output)

        assert output.shape == (batch_size, seq_len, expected_out_dim)

        output_np = np.array(output)
        assert not np.any(np.isnan(output_np)), "Batch output contains NaN"
        assert not np.any(np.isinf(output_np)), "Batch output contains Inf"

    def test_single_vs_batch_consistency(self, small_config):
        """Test that single sample matches batched processing."""
        # Create PyTorch reference for weight sharing
        torch_head = Head(**small_config)

        mlx_head = HeadMLX(**small_config)
        copy_weights_head(mlx_head, torch_head)

        dim = small_config['dim']
        seq_len = 32

        np.random.seed(666)
        np_x1 = np.random.randn(1, seq_len, dim).astype(np.float32)
        np_e1 = np.random.randn(1, seq_len, dim).astype(np.float32)
        np_x2 = np.random.randn(1, seq_len, dim).astype(np.float32)
        np_e2 = np.random.randn(1, seq_len, dim).astype(np.float32)

        # Process individually
        mlx_x1 = mx.array(np_x1)
        mlx_e1 = mx.array(np_e1)
        mlx_x2 = mx.array(np_x2)
        mlx_e2 = mx.array(np_e2)

        output1 = mlx_head(mlx_x1, mlx_e1)
        output2 = mlx_head(mlx_x2, mlx_e2)
        mx.eval(output1, output2)

        # Process as batch
        np_x_batch = np.concatenate([np_x1, np_x2], axis=0)
        np_e_batch = np.concatenate([np_e1, np_e2], axis=0)

        mlx_x_batch = mx.array(np_x_batch)
        mlx_e_batch = mx.array(np_e_batch)

        output_batch = mlx_head(mlx_x_batch, mlx_e_batch)
        mx.eval(output_batch)

        # Compare
        np.testing.assert_allclose(
            np.array(output1),
            np.array(output_batch)[0:1],
            rtol=1e-5,
            atol=1e-5,
            err_msg="Single sample doesn't match batched (sample 1)"
        )

        np.testing.assert_allclose(
            np.array(output2),
            np.array(output_batch)[1:2],
            rtol=1e-5,
            atol=1e-5,
            err_msg="Single sample doesn't match batched (sample 2)"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
