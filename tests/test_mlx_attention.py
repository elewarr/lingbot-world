"""Unit tests for MLX self-attention layer.

Tests that MLX implementation matches PyTorch reference within tolerance.

Note: PyTorch's flash_attention on MPS uses bfloat16 compute, while MLX uses
float32. Tests use float32 for both to ensure fair comparison.
"""

import numpy as np
import pytest

# Import frameworks
import mlx.core as mx
import torch
import torch.nn.functional as F

# Import MLX implementations to test
from wan.backends.mlx.attention import WanSelfAttentionMLX, WanCrossAttentionMLX
from wan.backends.mlx.rope import create_rope_freqs_mlx

# Import PyTorch reference
from wan.modules.model import WanSelfAttention, WanCrossAttention, rope_params, rope_apply


def pytorch_attention_float32(torch_attn, x, grid_sizes, freqs):
    """Compute PyTorch attention with float32 precision for fair comparison.
    
    PyTorch's flash_attention on MPS uses bfloat16 compute, which introduces
    ~1e-3 precision difference. This helper uses float32 for fair comparison.
    """
    b, s, _ = x.shape
    n, d = torch_attn.num_heads, torch_attn.head_dim
    
    with torch.no_grad():
        # Q, K, V projections
        q = torch_attn.q(x)
        k = torch_attn.k(x)
        v = torch_attn.v(x)
        
        # Apply norm if enabled
        if torch_attn.qk_norm:
            q = torch_attn.norm_q(q)
            k = torch_attn.norm_k(k)
        
        # Reshape to [B, L, N, D]
        q = q.view(b, s, n, d)
        k = k.view(b, s, n, d)
        v = v.view(b, s, n, d)
        
        # Apply RoPE
        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)
        
        # Transpose to [B, N, L, D] for SDPA
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        
        # SDPA with float32 precision
        scale = d ** -0.5
        attn_out = F.scaled_dot_product_attention(q_t, k_t, v_t, scale=scale)
        
        # Transpose back and flatten
        attn_out = attn_out.transpose(1, 2).contiguous()
        out = torch_attn.o(attn_out.flatten(2))
        
    return out


class TestWanSelfAttentionMLX:
    """Tests for WanSelfAttentionMLX comparing against PyTorch WanSelfAttention."""

    @pytest.fixture
    def default_config(self):
        """Default configuration matching WanModel."""
        return {
            'dim': 2048,
            'num_heads': 16,
            'window_size': (-1, -1),
            'qk_norm': True,
            'eps': 1e-6,
        }

    @pytest.fixture
    def small_config(self):
        """Smaller configuration for faster tests."""
        return {
            'dim': 256,
            'num_heads': 4,
            'window_size': (-1, -1),
            'qk_norm': True,
            'eps': 1e-6,
        }

    def test_initialization(self, default_config):
        """Test that WanSelfAttentionMLX initializes correctly."""
        attn = WanSelfAttentionMLX(**default_config)

        assert attn.dim == default_config['dim']
        assert attn.num_heads == default_config['num_heads']
        assert attn.head_dim == default_config['dim'] // default_config['num_heads']
        assert attn.window_size == default_config['window_size']
        assert attn.qk_norm == default_config['qk_norm']
        assert attn.eps == default_config['eps']

    def test_projection_shapes(self, small_config):
        """Test that Q, K, V, O projections have correct shapes."""
        attn = WanSelfAttentionMLX(**small_config)
        dim = small_config['dim']

        # Check weight shapes
        assert attn.q.weight.shape == (dim, dim)
        assert attn.k.weight.shape == (dim, dim)
        assert attn.v.weight.shape == (dim, dim)
        assert attn.o.weight.shape == (dim, dim)

    def test_qk_norm_initialization(self, small_config):
        """Test that QK normalization layers are initialized correctly."""
        # With qk_norm=True
        attn_with_norm = WanSelfAttentionMLX(**small_config)
        assert attn_with_norm.norm_q is not None
        assert attn_with_norm.norm_k is not None

        # With qk_norm=False
        config_no_norm = {**small_config, 'qk_norm': False}
        attn_no_norm = WanSelfAttentionMLX(**config_no_norm)
        assert attn_no_norm.norm_q is None
        assert attn_no_norm.norm_k is None

    def test_output_shape(self, small_config):
        """Test that forward pass produces correct output shape."""
        batch_size = 2
        dim = small_config['dim']
        num_heads = small_config['num_heads']
        head_dim = dim // num_heads

        # Grid: 4x4x4 = 64 positions
        f, h, w = 4, 4, 4
        seq_len = f * h * w

        # Create inputs
        np.random.seed(42)
        np_input = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
        grid_sizes = np.array([[f, h, w], [f, h, w]], dtype=np.int64)
        seq_lens = np.array([seq_len, seq_len], dtype=np.int64)

        # Create frequencies
        cos_tuple, sin_tuple = create_rope_freqs_mlx(1024, head_dim)

        # Create attention module
        attn = WanSelfAttentionMLX(**small_config)

        # Forward pass
        mlx_input = mx.array(np_input)
        mlx_grid = mx.array(grid_sizes)
        mlx_seq_lens = mx.array(seq_lens)
        output = attn(mlx_input, mlx_seq_lens, mlx_grid, cos_tuple, sin_tuple)
        mx.eval(output)

        # Check output shape
        assert output.shape == (batch_size, seq_len, dim)

    def test_pytorch_equivalence_basic(self, small_config):
        """Test that MLX output matches PyTorch within tolerance."""
        batch_size = 1
        dim = small_config['dim']
        num_heads = small_config['num_heads']
        head_dim = dim // num_heads

        # Grid: 4x4x4 = 64 positions
        f, h, w = 4, 4, 4
        seq_len = f * h * w

        # Create shared input
        np.random.seed(123)
        np_input = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
        grid_sizes_np = np.array([[f, h, w]], dtype=np.int64)
        seq_lens_np = np.array([seq_len], dtype=np.int64)

        # ---- PyTorch Reference ----
        torch_attn = WanSelfAttention(
            dim=dim,
            num_heads=num_heads,
            window_size=small_config['window_size'],
            qk_norm=small_config['qk_norm'],
            eps=small_config['eps'],
        )

        # Create PyTorch freqs
        torch_freqs = torch.cat([
            rope_params(1024, head_dim - 4 * (head_dim // 6)),
            rope_params(1024, 2 * (head_dim // 6)),
            rope_params(1024, 2 * (head_dim // 6))
        ], dim=1)

        torch_input = torch.from_numpy(np_input)
        torch_grid = torch.from_numpy(grid_sizes_np)

        # Use float32 reference for fair comparison
        torch_output = pytorch_attention_float32(torch_attn, torch_input, torch_grid, torch_freqs)
        torch_output_np = torch_output.numpy()

        # ---- MLX Implementation ----
        mlx_attn = WanSelfAttentionMLX(**small_config)

        # Copy weights from PyTorch to MLX
        # MLX Linear uses same weight layout as PyTorch (no transpose needed)
        mlx_attn.q.weight = mx.array(torch_attn.q.weight.detach().numpy())
        mlx_attn.q.bias = mx.array(torch_attn.q.bias.detach().numpy())
        mlx_attn.k.weight = mx.array(torch_attn.k.weight.detach().numpy())
        mlx_attn.k.bias = mx.array(torch_attn.k.bias.detach().numpy())
        mlx_attn.v.weight = mx.array(torch_attn.v.weight.detach().numpy())
        mlx_attn.v.bias = mx.array(torch_attn.v.bias.detach().numpy())
        mlx_attn.o.weight = mx.array(torch_attn.o.weight.detach().numpy())
        mlx_attn.o.bias = mx.array(torch_attn.o.bias.detach().numpy())

        # Copy norm weights
        if small_config['qk_norm']:
            mlx_attn.norm_q.weight = mx.array(torch_attn.norm_q.weight.detach().numpy())
            mlx_attn.norm_k.weight = mx.array(torch_attn.norm_k.weight.detach().numpy())

        # Create MLX freqs
        cos_tuple, sin_tuple = create_rope_freqs_mlx(1024, head_dim)

        mlx_input = mx.array(np_input)
        mlx_grid = mx.array(grid_sizes_np)
        mlx_seq_lens = mx.array(seq_lens_np)
        mlx_output = mlx_attn(mlx_input, mlx_seq_lens, mlx_grid, cos_tuple, sin_tuple)
        mx.eval(mlx_output)
        mlx_output_np = np.array(mlx_output)

        # Compare outputs
        np.testing.assert_allclose(
            mlx_output_np,
            torch_output_np,
            rtol=1e-4,
            atol=1e-4,
            err_msg="MLX output doesn't match PyTorch within tolerance"
        )

    def test_wanmodel_config(self):
        """Test with the exact WanModel configuration (dim=2048, num_heads=16)."""
        batch_size = 1
        dim = 2048
        num_heads = 16
        head_dim = dim // num_heads  # 128

        # Smaller grid for faster test
        f, h, w = 4, 4, 4  # 64 positions
        seq_len = f * h * w

        config = {
            'dim': dim,
            'num_heads': num_heads,
            'window_size': (-1, -1),
            'qk_norm': True,
            'eps': 1e-6,
        }

        # Create shared input
        np.random.seed(456)
        np_input = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
        grid_sizes_np = np.array([[f, h, w]], dtype=np.int64)
        seq_lens_np = np.array([seq_len], dtype=np.int64)

        # ---- PyTorch Reference ----
        torch_attn = WanSelfAttention(**config)
        torch_freqs = torch.cat([
            rope_params(1024, head_dim - 4 * (head_dim // 6)),
            rope_params(1024, 2 * (head_dim // 6)),
            rope_params(1024, 2 * (head_dim // 6))
        ], dim=1)

        torch_input = torch.from_numpy(np_input)
        torch_grid = torch.from_numpy(grid_sizes_np)

        # Use float32 reference for fair comparison
        torch_output = pytorch_attention_float32(torch_attn, torch_input, torch_grid, torch_freqs)
        torch_output_np = torch_output.numpy()

        # ---- MLX Implementation ----
        mlx_attn = WanSelfAttentionMLX(**config)

        # Copy weights from PyTorch to MLX
        mlx_attn.q.weight = mx.array(torch_attn.q.weight.detach().numpy())
        mlx_attn.q.bias = mx.array(torch_attn.q.bias.detach().numpy())
        mlx_attn.k.weight = mx.array(torch_attn.k.weight.detach().numpy())
        mlx_attn.k.bias = mx.array(torch_attn.k.bias.detach().numpy())
        mlx_attn.v.weight = mx.array(torch_attn.v.weight.detach().numpy())
        mlx_attn.v.bias = mx.array(torch_attn.v.bias.detach().numpy())
        mlx_attn.o.weight = mx.array(torch_attn.o.weight.detach().numpy())
        mlx_attn.o.bias = mx.array(torch_attn.o.bias.detach().numpy())
        mlx_attn.norm_q.weight = mx.array(torch_attn.norm_q.weight.detach().numpy())
        mlx_attn.norm_k.weight = mx.array(torch_attn.norm_k.weight.detach().numpy())

        cos_tuple, sin_tuple = create_rope_freqs_mlx(1024, head_dim)

        mlx_input = mx.array(np_input)
        mlx_grid = mx.array(grid_sizes_np)
        mlx_seq_lens = mx.array(seq_lens_np)
        mlx_output = mlx_attn(mlx_input, mlx_seq_lens, mlx_grid, cos_tuple, sin_tuple)
        mx.eval(mlx_output)
        mlx_output_np = np.array(mlx_output)

        # Compare outputs
        np.testing.assert_allclose(
            mlx_output_np,
            torch_output_np,
            rtol=1e-4,
            atol=1e-4,
            err_msg="MLX doesn't match PyTorch for WanModel config"
        )

    def test_batch_processing(self, small_config):
        """Test that batched inputs are processed correctly."""
        batch_size = 4
        dim = small_config['dim']
        num_heads = small_config['num_heads']
        head_dim = dim // num_heads

        # Same grid for all samples
        f, h, w = 4, 4, 4
        seq_len = f * h * w

        # Create inputs
        np.random.seed(789)
        np_input = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
        grid_sizes = np.array([[f, h, w]] * batch_size, dtype=np.int64)
        seq_lens = np.array([seq_len] * batch_size, dtype=np.int64)

        # Create attention module
        attn = WanSelfAttentionMLX(**small_config)

        # Forward pass
        cos_tuple, sin_tuple = create_rope_freqs_mlx(1024, head_dim)
        mlx_input = mx.array(np_input)
        mlx_grid = mx.array(grid_sizes)
        mlx_seq_lens = mx.array(seq_lens)
        output = attn(mlx_input, mlx_seq_lens, mlx_grid, cos_tuple, sin_tuple)
        mx.eval(output)

        # Check output shape
        assert output.shape == (batch_size, seq_len, dim)

        # Check no NaN/Inf
        output_np = np.array(output)
        assert not np.any(np.isnan(output_np)), "Output contains NaN"
        assert not np.any(np.isinf(output_np)), "Output contains Inf"

    def test_without_qk_norm(self, small_config):
        """Test attention layer without QK normalization."""
        config_no_norm = {**small_config, 'qk_norm': False}
        batch_size = 1
        dim = small_config['dim']
        num_heads = small_config['num_heads']
        head_dim = dim // num_heads

        f, h, w = 4, 4, 4
        seq_len = f * h * w

        # Create inputs
        np.random.seed(111)
        np_input = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
        grid_sizes_np = np.array([[f, h, w]], dtype=np.int64)
        seq_lens_np = np.array([seq_len], dtype=np.int64)

        # ---- PyTorch Reference ----
        torch_attn = WanSelfAttention(**config_no_norm)
        torch_freqs = torch.cat([
            rope_params(1024, head_dim - 4 * (head_dim // 6)),
            rope_params(1024, 2 * (head_dim // 6)),
            rope_params(1024, 2 * (head_dim // 6))
        ], dim=1)

        torch_input = torch.from_numpy(np_input)
        torch_grid = torch.from_numpy(grid_sizes_np)

        # Use float32 reference for fair comparison
        torch_output = pytorch_attention_float32(torch_attn, torch_input, torch_grid, torch_freqs)
        torch_output_np = torch_output.numpy()

        # ---- MLX Implementation ----
        mlx_attn = WanSelfAttentionMLX(**config_no_norm)

        # Copy weights
        mlx_attn.q.weight = mx.array(torch_attn.q.weight.detach().numpy())
        mlx_attn.q.bias = mx.array(torch_attn.q.bias.detach().numpy())
        mlx_attn.k.weight = mx.array(torch_attn.k.weight.detach().numpy())
        mlx_attn.k.bias = mx.array(torch_attn.k.bias.detach().numpy())
        mlx_attn.v.weight = mx.array(torch_attn.v.weight.detach().numpy())
        mlx_attn.v.bias = mx.array(torch_attn.v.bias.detach().numpy())
        mlx_attn.o.weight = mx.array(torch_attn.o.weight.detach().numpy())
        mlx_attn.o.bias = mx.array(torch_attn.o.bias.detach().numpy())

        cos_tuple, sin_tuple = create_rope_freqs_mlx(1024, head_dim)

        mlx_input = mx.array(np_input)
        mlx_grid = mx.array(grid_sizes_np)
        mlx_seq_lens = mx.array(seq_lens_np)
        mlx_output = mlx_attn(mlx_input, mlx_seq_lens, mlx_grid, cos_tuple, sin_tuple)
        mx.eval(mlx_output)
        mlx_output_np = np.array(mlx_output)

        # Compare outputs
        np.testing.assert_allclose(
            mlx_output_np,
            torch_output_np,
            rtol=1e-4,
            atol=1e-4,
            err_msg="MLX without qk_norm doesn't match PyTorch"
        )

    def test_variable_sequence_lengths(self, small_config):
        """Test with variable-length sequences in a batch."""
        batch_size = 2
        dim = small_config['dim']
        num_heads = small_config['num_heads']
        head_dim = dim // num_heads

        # Different grid sizes
        grids = [(4, 4, 4), (4, 4, 2)]  # 64 and 32 positions
        max_seq_len = max(f * h * w for f, h, w in grids)

        # Create padded input
        np.random.seed(222)
        np_input = np.random.randn(batch_size, max_seq_len, dim).astype(np.float32)
        grid_sizes_np = np.array(grids, dtype=np.int64)
        seq_lens_np = np.array([f * h * w for f, h, w in grids], dtype=np.int64)

        # Create attention module
        attn = WanSelfAttentionMLX(**small_config)

        # Forward pass
        cos_tuple, sin_tuple = create_rope_freqs_mlx(1024, head_dim)
        mlx_input = mx.array(np_input)
        mlx_grid = mx.array(grid_sizes_np)
        mlx_seq_lens = mx.array(seq_lens_np)
        output = attn(mlx_input, mlx_seq_lens, mlx_grid, cos_tuple, sin_tuple)
        mx.eval(output)

        # Check output shape
        assert output.shape == (batch_size, max_seq_len, dim)

        # Check no NaN/Inf
        output_np = np.array(output)
        assert not np.any(np.isnan(output_np)), "Output contains NaN"
        assert not np.any(np.isinf(output_np)), "Output contains Inf"

    def test_numerical_stability(self, small_config):
        """Test numerical stability with various input magnitudes."""
        batch_size = 1
        dim = small_config['dim']
        num_heads = small_config['num_heads']
        head_dim = dim // num_heads

        f, h, w = 4, 4, 4
        seq_len = f * h * w

        attn = WanSelfAttentionMLX(**small_config)
        cos_tuple, sin_tuple = create_rope_freqs_mlx(1024, head_dim)

        # Test with small values
        np.random.seed(333)
        small_input = np.random.randn(batch_size, seq_len, dim).astype(np.float32) * 1e-3
        grid_sizes = np.array([[f, h, w]], dtype=np.int64)
        seq_lens = np.array([seq_len], dtype=np.int64)

        output_small = attn(
            mx.array(small_input),
            mx.array(seq_lens),
            mx.array(grid_sizes),
            cos_tuple, sin_tuple
        )
        mx.eval(output_small)
        output_small_np = np.array(output_small)

        assert not np.any(np.isnan(output_small_np)), "Small input produced NaN"
        assert not np.any(np.isinf(output_small_np)), "Small input produced Inf"

        # Test with large values
        large_input = np.random.randn(batch_size, seq_len, dim).astype(np.float32) * 100
        output_large = attn(
            mx.array(large_input),
            mx.array(seq_lens),
            mx.array(grid_sizes),
            cos_tuple, sin_tuple
        )
        mx.eval(output_large)
        output_large_np = np.array(output_large)

        assert not np.any(np.isnan(output_large_np)), "Large input produced NaN"
        assert not np.any(np.isinf(output_large_np)), "Large input produced Inf"

    def test_dim_heads_validation(self):
        """Test that dim must be divisible by num_heads."""
        with pytest.raises(AssertionError):
            WanSelfAttentionMLX(dim=100, num_heads=16)  # 100 not divisible by 16

    def test_deterministic_output(self, small_config):
        """Test that the same input produces the same output."""
        batch_size = 1
        dim = small_config['dim']
        num_heads = small_config['num_heads']
        head_dim = dim // num_heads

        f, h, w = 4, 4, 4
        seq_len = f * h * w

        # Create input
        np.random.seed(444)
        np_input = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
        grid_sizes = np.array([[f, h, w]], dtype=np.int64)
        seq_lens = np.array([seq_len], dtype=np.int64)

        attn = WanSelfAttentionMLX(**small_config)
        cos_tuple, sin_tuple = create_rope_freqs_mlx(1024, head_dim)

        # Run twice
        mlx_input = mx.array(np_input)
        mlx_grid = mx.array(grid_sizes)
        mlx_seq_lens = mx.array(seq_lens)

        output1 = attn(mlx_input, mlx_seq_lens, mlx_grid, cos_tuple, sin_tuple)
        mx.eval(output1)
        output2 = attn(mlx_input, mlx_seq_lens, mlx_grid, cos_tuple, sin_tuple)
        mx.eval(output2)

        # Should be identical
        np.testing.assert_array_equal(
            np.array(output1),
            np.array(output2),
            err_msg="Same input produced different outputs"
        )


class TestAttentionMaskCreation:
    """Tests for attention mask creation logic."""

    def test_no_mask_when_all_same_length(self):
        """Test that no mask is created when all sequences are same length."""
        attn = WanSelfAttentionMLX(dim=256, num_heads=4)
        seq_lens = mx.array([64, 64, 64], dtype=mx.int64)
        mask = attn._create_attention_mask(seq_lens, 64)

        assert mask is None, "Should not create mask when all sequences same length"

    def test_mask_shape_with_padding(self):
        """Test mask shape when sequences have different lengths."""
        attn = WanSelfAttentionMLX(dim=256, num_heads=4)
        seq_lens = mx.array([64, 32], dtype=mx.int64)
        max_seq_len = 64
        mask = attn._create_attention_mask(seq_lens, max_seq_len)

        assert mask is not None, "Should create mask when sequences differ"
        assert mask.shape == (2, 1, 1, max_seq_len), f"Expected shape (2, 1, 1, 64), got {mask.shape}"

    def test_mask_values(self):
        """Test that mask has correct values (0 for valid, -inf for padding)."""
        attn = WanSelfAttentionMLX(dim=256, num_heads=4)
        seq_lens = mx.array([4, 2], dtype=mx.int64)
        max_seq_len = 4
        mask = attn._create_attention_mask(seq_lens, max_seq_len)
        mx.eval(mask)
        mask_np = np.array(mask)

        # First sample: all 4 positions valid
        assert np.all(mask_np[0, 0, 0, :4] == 0), "First sample should have all valid positions"

        # Second sample: first 2 valid, last 2 masked
        assert np.all(mask_np[1, 0, 0, :2] == 0), "Second sample first 2 should be valid"
        assert np.all(np.isinf(mask_np[1, 0, 0, 2:])), "Second sample last 2 should be -inf"


def pytorch_cross_attention_float32(torch_attn, x, context, context_lens):
    """Compute PyTorch cross-attention with float32 precision for fair comparison.
    
    PyTorch's flash_attention on MPS uses bfloat16 compute, which introduces
    ~1e-3 precision difference. This helper uses float32 for fair comparison.
    """
    b = x.shape[0]
    n, d = torch_attn.num_heads, torch_attn.head_dim
    
    with torch.no_grad():
        # Q from x
        q = torch_attn.q(x)
        if torch_attn.qk_norm:
            q = torch_attn.norm_q(q)
        
        # K, V from context
        k = torch_attn.k(context)
        if torch_attn.qk_norm:
            k = torch_attn.norm_k(k)
        v = torch_attn.v(context)
        
        # Reshape to [B, L, N, D]
        q = q.view(b, -1, n, d)
        k = k.view(b, -1, n, d)
        v = v.view(b, -1, n, d)
        
        # Transpose to [B, N, L, D] for SDPA
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        
        # Create attention mask for variable context lengths if needed
        attn_mask = None
        if context_lens is not None:
            max_ctx_len = k_t.shape[2]
            # Create mask for each sample
            masks = []
            for i in range(b):
                ctx_len = int(context_lens[i].item())
                # Create row mask: True for valid, False for padding
                row_mask = torch.arange(max_ctx_len) < ctx_len
                masks.append(row_mask)
            # Stack to [B, L2] and expand to [B, 1, 1, L2]
            key_mask = torch.stack(masks, dim=0).unsqueeze(1).unsqueeze(2)
            # Convert to additive mask
            attn_mask = torch.where(key_mask, 0.0, float('-inf'))
        
        # SDPA with float32 precision
        scale = d ** -0.5
        attn_out = F.scaled_dot_product_attention(q_t, k_t, v_t, scale=scale, attn_mask=attn_mask)
        
        # Transpose back and flatten
        attn_out = attn_out.transpose(1, 2).contiguous()
        out = torch_attn.o(attn_out.flatten(2))
        
    return out


class TestWanCrossAttentionMLX:
    """Tests for WanCrossAttentionMLX comparing against PyTorch WanCrossAttention."""

    @pytest.fixture
    def default_config(self):
        """Default configuration matching WanModel."""
        return {
            'dim': 2048,
            'num_heads': 16,
            'window_size': (-1, -1),
            'qk_norm': True,
            'eps': 1e-6,
        }

    @pytest.fixture
    def small_config(self):
        """Smaller configuration for faster tests."""
        return {
            'dim': 256,
            'num_heads': 4,
            'window_size': (-1, -1),
            'qk_norm': True,
            'eps': 1e-6,
        }

    def test_initialization(self, default_config):
        """Test that WanCrossAttentionMLX initializes correctly."""
        attn = WanCrossAttentionMLX(**default_config)

        assert attn.dim == default_config['dim']
        assert attn.num_heads == default_config['num_heads']
        assert attn.head_dim == default_config['dim'] // default_config['num_heads']

    def test_inherits_from_self_attention(self, small_config):
        """Test that WanCrossAttentionMLX inherits from WanSelfAttentionMLX."""
        attn = WanCrossAttentionMLX(**small_config)
        assert isinstance(attn, WanSelfAttentionMLX)

    def test_output_shape(self, small_config):
        """Test that forward pass produces correct output shape."""
        batch_size = 2
        dim = small_config['dim']
        hidden_seq_len = 64  # x sequence length
        context_seq_len = 128  # context sequence length (text embeddings)

        # Create inputs
        np.random.seed(42)
        np_x = np.random.randn(batch_size, hidden_seq_len, dim).astype(np.float32)
        np_context = np.random.randn(batch_size, context_seq_len, dim).astype(np.float32)
        context_lens = np.array([context_seq_len, context_seq_len], dtype=np.int64)

        # Create attention module
        attn = WanCrossAttentionMLX(**small_config)

        # Forward pass
        mlx_x = mx.array(np_x)
        mlx_context = mx.array(np_context)
        mlx_context_lens = mx.array(context_lens)
        output = attn(mlx_x, mlx_context, mlx_context_lens)
        mx.eval(output)

        # Output shape should match x's sequence length (not context's)
        assert output.shape == (batch_size, hidden_seq_len, dim)

    def test_pytorch_equivalence_basic(self, small_config):
        """Test that MLX output matches PyTorch within tolerance."""
        batch_size = 1
        dim = small_config['dim']
        hidden_seq_len = 64
        context_seq_len = 128

        # Create shared inputs
        np.random.seed(123)
        np_x = np.random.randn(batch_size, hidden_seq_len, dim).astype(np.float32)
        np_context = np.random.randn(batch_size, context_seq_len, dim).astype(np.float32)

        # ---- PyTorch Reference ----
        torch_attn = WanCrossAttention(
            dim=dim,
            num_heads=small_config['num_heads'],
            window_size=small_config['window_size'],
            qk_norm=small_config['qk_norm'],
            eps=small_config['eps'],
        )

        torch_x = torch.from_numpy(np_x)
        torch_context = torch.from_numpy(np_context)

        # Use float32 reference (no context_lens masking for this test)
        torch_output = pytorch_cross_attention_float32(torch_attn, torch_x, torch_context, None)
        torch_output_np = torch_output.numpy()

        # ---- MLX Implementation ----
        mlx_attn = WanCrossAttentionMLX(**small_config)

        # Copy weights from PyTorch to MLX
        mlx_attn.q.weight = mx.array(torch_attn.q.weight.detach().numpy())
        mlx_attn.q.bias = mx.array(torch_attn.q.bias.detach().numpy())
        mlx_attn.k.weight = mx.array(torch_attn.k.weight.detach().numpy())
        mlx_attn.k.bias = mx.array(torch_attn.k.bias.detach().numpy())
        mlx_attn.v.weight = mx.array(torch_attn.v.weight.detach().numpy())
        mlx_attn.v.bias = mx.array(torch_attn.v.bias.detach().numpy())
        mlx_attn.o.weight = mx.array(torch_attn.o.weight.detach().numpy())
        mlx_attn.o.bias = mx.array(torch_attn.o.bias.detach().numpy())

        # Copy norm weights
        if small_config['qk_norm']:
            mlx_attn.norm_q.weight = mx.array(torch_attn.norm_q.weight.detach().numpy())
            mlx_attn.norm_k.weight = mx.array(torch_attn.norm_k.weight.detach().numpy())

        mlx_x = mx.array(np_x)
        mlx_context = mx.array(np_context)
        mlx_output = mlx_attn(mlx_x, mlx_context, None)
        mx.eval(mlx_output)
        mlx_output_np = np.array(mlx_output)

        # Compare outputs
        np.testing.assert_allclose(
            mlx_output_np,
            torch_output_np,
            rtol=1e-4,
            atol=1e-4,
            err_msg="MLX cross-attention output doesn't match PyTorch within tolerance"
        )

    def test_wanmodel_config(self):
        """Test with the exact WanModel configuration (dim=2048, num_heads=16)."""
        batch_size = 1
        dim = 2048
        num_heads = 16
        hidden_seq_len = 64
        context_seq_len = 128  # text_len in WanModel

        config = {
            'dim': dim,
            'num_heads': num_heads,
            'window_size': (-1, -1),
            'qk_norm': True,
            'eps': 1e-6,
        }

        # Create shared inputs
        np.random.seed(456)
        np_x = np.random.randn(batch_size, hidden_seq_len, dim).astype(np.float32)
        np_context = np.random.randn(batch_size, context_seq_len, dim).astype(np.float32)

        # ---- PyTorch Reference ----
        torch_attn = WanCrossAttention(**config)
        torch_x = torch.from_numpy(np_x)
        torch_context = torch.from_numpy(np_context)

        torch_output = pytorch_cross_attention_float32(torch_attn, torch_x, torch_context, None)
        torch_output_np = torch_output.numpy()

        # ---- MLX Implementation ----
        mlx_attn = WanCrossAttentionMLX(**config)

        # Copy weights
        mlx_attn.q.weight = mx.array(torch_attn.q.weight.detach().numpy())
        mlx_attn.q.bias = mx.array(torch_attn.q.bias.detach().numpy())
        mlx_attn.k.weight = mx.array(torch_attn.k.weight.detach().numpy())
        mlx_attn.k.bias = mx.array(torch_attn.k.bias.detach().numpy())
        mlx_attn.v.weight = mx.array(torch_attn.v.weight.detach().numpy())
        mlx_attn.v.bias = mx.array(torch_attn.v.bias.detach().numpy())
        mlx_attn.o.weight = mx.array(torch_attn.o.weight.detach().numpy())
        mlx_attn.o.bias = mx.array(torch_attn.o.bias.detach().numpy())
        mlx_attn.norm_q.weight = mx.array(torch_attn.norm_q.weight.detach().numpy())
        mlx_attn.norm_k.weight = mx.array(torch_attn.norm_k.weight.detach().numpy())

        mlx_x = mx.array(np_x)
        mlx_context = mx.array(np_context)
        mlx_output = mlx_attn(mlx_x, mlx_context, None)
        mx.eval(mlx_output)
        mlx_output_np = np.array(mlx_output)

        # Compare outputs
        np.testing.assert_allclose(
            mlx_output_np,
            torch_output_np,
            rtol=1e-4,
            atol=1e-4,
            err_msg="MLX doesn't match PyTorch for WanModel cross-attention config"
        )

    def test_text_conditioning_scenario(self, small_config):
        """Test typical text conditioning scenario (different x and context lengths)."""
        batch_size = 2
        dim = small_config['dim']
        hidden_seq_len = 256  # Larger hidden state (video frames)
        context_seq_len = 77  # Typical text encoder output length

        # Create inputs
        np.random.seed(789)
        np_x = np.random.randn(batch_size, hidden_seq_len, dim).astype(np.float32)
        np_context = np.random.randn(batch_size, context_seq_len, dim).astype(np.float32)

        attn = WanCrossAttentionMLX(**small_config)

        mlx_x = mx.array(np_x)
        mlx_context = mx.array(np_context)
        output = attn(mlx_x, mlx_context, None)
        mx.eval(output)

        # Output should match x's length, not context's
        assert output.shape == (batch_size, hidden_seq_len, dim)
        
        output_np = np.array(output)
        assert not np.any(np.isnan(output_np)), "Output contains NaN"
        assert not np.any(np.isinf(output_np)), "Output contains Inf"

    def test_variable_context_lengths(self, small_config):
        """Test with variable-length context sequences in a batch."""
        batch_size = 2
        dim = small_config['dim']
        hidden_seq_len = 64
        max_context_len = 128

        # Different context lengths
        context_lens = np.array([128, 64], dtype=np.int64)

        # Create inputs
        np.random.seed(111)
        np_x = np.random.randn(batch_size, hidden_seq_len, dim).astype(np.float32)
        np_context = np.random.randn(batch_size, max_context_len, dim).astype(np.float32)

        attn = WanCrossAttentionMLX(**small_config)

        mlx_x = mx.array(np_x)
        mlx_context = mx.array(np_context)
        mlx_context_lens = mx.array(context_lens)
        output = attn(mlx_x, mlx_context, mlx_context_lens)
        mx.eval(output)

        assert output.shape == (batch_size, hidden_seq_len, dim)
        
        output_np = np.array(output)
        assert not np.any(np.isnan(output_np)), "Output contains NaN"
        assert not np.any(np.isinf(output_np)), "Output contains Inf"

    def test_variable_context_pytorch_equivalence(self, small_config):
        """Test that variable context lengths match PyTorch."""
        batch_size = 2
        dim = small_config['dim']
        hidden_seq_len = 64
        max_context_len = 128

        # Different context lengths
        context_lens_np = np.array([128, 64], dtype=np.int64)

        # Create inputs
        np.random.seed(222)
        np_x = np.random.randn(batch_size, hidden_seq_len, dim).astype(np.float32)
        np_context = np.random.randn(batch_size, max_context_len, dim).astype(np.float32)

        # ---- PyTorch Reference ----
        torch_attn = WanCrossAttention(**small_config)
        torch_x = torch.from_numpy(np_x)
        torch_context = torch.from_numpy(np_context)
        torch_context_lens = torch.from_numpy(context_lens_np)

        torch_output = pytorch_cross_attention_float32(
            torch_attn, torch_x, torch_context, torch_context_lens
        )
        torch_output_np = torch_output.numpy()

        # ---- MLX Implementation ----
        mlx_attn = WanCrossAttentionMLX(**small_config)

        # Copy weights
        mlx_attn.q.weight = mx.array(torch_attn.q.weight.detach().numpy())
        mlx_attn.q.bias = mx.array(torch_attn.q.bias.detach().numpy())
        mlx_attn.k.weight = mx.array(torch_attn.k.weight.detach().numpy())
        mlx_attn.k.bias = mx.array(torch_attn.k.bias.detach().numpy())
        mlx_attn.v.weight = mx.array(torch_attn.v.weight.detach().numpy())
        mlx_attn.v.bias = mx.array(torch_attn.v.bias.detach().numpy())
        mlx_attn.o.weight = mx.array(torch_attn.o.weight.detach().numpy())
        mlx_attn.o.bias = mx.array(torch_attn.o.bias.detach().numpy())
        if small_config['qk_norm']:
            mlx_attn.norm_q.weight = mx.array(torch_attn.norm_q.weight.detach().numpy())
            mlx_attn.norm_k.weight = mx.array(torch_attn.norm_k.weight.detach().numpy())

        mlx_x = mx.array(np_x)
        mlx_context = mx.array(np_context)
        mlx_context_lens = mx.array(context_lens_np)
        mlx_output = mlx_attn(mlx_x, mlx_context, mlx_context_lens)
        mx.eval(mlx_output)
        mlx_output_np = np.array(mlx_output)

        # Compare outputs
        np.testing.assert_allclose(
            mlx_output_np,
            torch_output_np,
            rtol=1e-4,
            atol=1e-4,
            err_msg="MLX cross-attention with variable context lens doesn't match PyTorch"
        )

    def test_without_qk_norm(self, small_config):
        """Test cross-attention layer without QK normalization."""
        config_no_norm = {**small_config, 'qk_norm': False}
        batch_size = 1
        dim = small_config['dim']
        hidden_seq_len = 64
        context_seq_len = 128

        # Create inputs
        np.random.seed(333)
        np_x = np.random.randn(batch_size, hidden_seq_len, dim).astype(np.float32)
        np_context = np.random.randn(batch_size, context_seq_len, dim).astype(np.float32)

        # ---- PyTorch Reference ----
        torch_attn = WanCrossAttention(**config_no_norm)
        torch_x = torch.from_numpy(np_x)
        torch_context = torch.from_numpy(np_context)

        torch_output = pytorch_cross_attention_float32(torch_attn, torch_x, torch_context, None)
        torch_output_np = torch_output.numpy()

        # ---- MLX Implementation ----
        mlx_attn = WanCrossAttentionMLX(**config_no_norm)

        # Copy weights
        mlx_attn.q.weight = mx.array(torch_attn.q.weight.detach().numpy())
        mlx_attn.q.bias = mx.array(torch_attn.q.bias.detach().numpy())
        mlx_attn.k.weight = mx.array(torch_attn.k.weight.detach().numpy())
        mlx_attn.k.bias = mx.array(torch_attn.k.bias.detach().numpy())
        mlx_attn.v.weight = mx.array(torch_attn.v.weight.detach().numpy())
        mlx_attn.v.bias = mx.array(torch_attn.v.bias.detach().numpy())
        mlx_attn.o.weight = mx.array(torch_attn.o.weight.detach().numpy())
        mlx_attn.o.bias = mx.array(torch_attn.o.bias.detach().numpy())

        mlx_x = mx.array(np_x)
        mlx_context = mx.array(np_context)
        mlx_output = mlx_attn(mlx_x, mlx_context, None)
        mx.eval(mlx_output)
        mlx_output_np = np.array(mlx_output)

        # Compare outputs
        np.testing.assert_allclose(
            mlx_output_np,
            torch_output_np,
            rtol=1e-4,
            atol=1e-4,
            err_msg="MLX cross-attention without qk_norm doesn't match PyTorch"
        )

    def test_numerical_stability(self, small_config):
        """Test numerical stability with various input magnitudes."""
        batch_size = 1
        dim = small_config['dim']
        hidden_seq_len = 64
        context_seq_len = 128

        attn = WanCrossAttentionMLX(**small_config)

        # Test with small values
        np.random.seed(444)
        small_x = np.random.randn(batch_size, hidden_seq_len, dim).astype(np.float32) * 1e-3
        small_ctx = np.random.randn(batch_size, context_seq_len, dim).astype(np.float32) * 1e-3

        output_small = attn(mx.array(small_x), mx.array(small_ctx), None)
        mx.eval(output_small)
        output_small_np = np.array(output_small)

        assert not np.any(np.isnan(output_small_np)), "Small input produced NaN"
        assert not np.any(np.isinf(output_small_np)), "Small input produced Inf"

        # Test with large values
        large_x = np.random.randn(batch_size, hidden_seq_len, dim).astype(np.float32) * 100
        large_ctx = np.random.randn(batch_size, context_seq_len, dim).astype(np.float32) * 100

        output_large = attn(mx.array(large_x), mx.array(large_ctx), None)
        mx.eval(output_large)
        output_large_np = np.array(output_large)

        assert not np.any(np.isnan(output_large_np)), "Large input produced NaN"
        assert not np.any(np.isinf(output_large_np)), "Large input produced Inf"

    def test_deterministic_output(self, small_config):
        """Test that the same input produces the same output."""
        batch_size = 1
        dim = small_config['dim']
        hidden_seq_len = 64
        context_seq_len = 128

        np.random.seed(555)
        np_x = np.random.randn(batch_size, hidden_seq_len, dim).astype(np.float32)
        np_context = np.random.randn(batch_size, context_seq_len, dim).astype(np.float32)

        attn = WanCrossAttentionMLX(**small_config)

        mlx_x = mx.array(np_x)
        mlx_context = mx.array(np_context)

        output1 = attn(mlx_x, mlx_context, None)
        mx.eval(output1)
        output2 = attn(mlx_x, mlx_context, None)
        mx.eval(output2)

        np.testing.assert_array_equal(
            np.array(output1),
            np.array(output2),
            err_msg="Same input produced different outputs"
        )


class TestCrossAttentionMaskCreation:
    """Tests for cross-attention mask creation logic."""

    def test_no_mask_when_context_lens_none(self):
        """Test that no mask is created when context_lens is None."""
        attn = WanCrossAttentionMLX(dim=256, num_heads=4)
        mask = attn._create_cross_attention_mask(None, 64, 128)
        assert mask is None, "Should not create mask when context_lens is None"

    def test_no_mask_when_all_same_length(self):
        """Test that no mask is created when all contexts are same length."""
        attn = WanCrossAttentionMLX(dim=256, num_heads=4)
        context_lens = mx.array([128, 128, 128], dtype=mx.int64)
        mask = attn._create_cross_attention_mask(context_lens, 64, 128)
        assert mask is None, "Should not create mask when all contexts same length"

    def test_mask_shape_with_variable_lengths(self):
        """Test mask shape when contexts have different lengths."""
        attn = WanCrossAttentionMLX(dim=256, num_heads=4)
        context_lens = mx.array([128, 64], dtype=mx.int64)
        mask = attn._create_cross_attention_mask(context_lens, 64, 128)

        assert mask is not None, "Should create mask when contexts differ"
        # Shape: [B, 1, 1, L2] for cross-attention
        assert mask.shape == (2, 1, 1, 128), f"Expected shape (2, 1, 1, 128), got {mask.shape}"

    def test_mask_values(self):
        """Test that mask has correct values (0 for valid, -inf for padding)."""
        attn = WanCrossAttentionMLX(dim=256, num_heads=4)
        context_lens = mx.array([4, 2], dtype=mx.int64)
        mask = attn._create_cross_attention_mask(context_lens, 8, 4)
        mx.eval(mask)
        mask_np = np.array(mask)

        # First sample: all 4 context positions valid
        assert np.all(mask_np[0, 0, 0, :4] == 0), "First sample should have all valid positions"

        # Second sample: first 2 valid, last 2 masked
        assert np.all(mask_np[1, 0, 0, :2] == 0), "Second sample first 2 should be valid"
        assert np.all(np.isinf(mask_np[1, 0, 0, 2:])), "Second sample last 2 should be -inf"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
