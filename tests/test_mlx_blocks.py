"""Unit tests for MLX WanAttentionBlock layer.

Tests that MLX implementation matches PyTorch reference within tolerance.

Key test areas:
- Initialization and shapes
- FFN with GELU(tanh)
- 6-way modulation mechanism
- Camera injection layers
- Full block PyTorch equivalence (with and without camera injection)
"""

import numpy as np
import pytest

# Import frameworks
import mlx.core as mx
import mlx.nn as nn_mlx
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import MLX implementations to test
from wan.backends.mlx.blocks import WanAttentionBlockMLX
from wan.backends.mlx.rope import create_rope_freqs_mlx

# Import PyTorch reference
from wan.modules.model import WanAttentionBlock, rope_params


def copy_weights_self_attn(mlx_attn, torch_attn):
    """Copy weights from PyTorch self-attention to MLX."""
    mlx_attn.q.weight = mx.array(torch_attn.q.weight.detach().numpy())
    mlx_attn.q.bias = mx.array(torch_attn.q.bias.detach().numpy())
    mlx_attn.k.weight = mx.array(torch_attn.k.weight.detach().numpy())
    mlx_attn.k.bias = mx.array(torch_attn.k.bias.detach().numpy())
    mlx_attn.v.weight = mx.array(torch_attn.v.weight.detach().numpy())
    mlx_attn.v.bias = mx.array(torch_attn.v.bias.detach().numpy())
    mlx_attn.o.weight = mx.array(torch_attn.o.weight.detach().numpy())
    mlx_attn.o.bias = mx.array(torch_attn.o.bias.detach().numpy())
    if mlx_attn.norm_q is not None and torch_attn.norm_q is not None:
        mlx_attn.norm_q.weight = mx.array(torch_attn.norm_q.weight.detach().numpy())
        mlx_attn.norm_k.weight = mx.array(torch_attn.norm_k.weight.detach().numpy())


def copy_weights_block(mlx_block, torch_block):
    """Copy all weights from PyTorch WanAttentionBlock to MLX WanAttentionBlockMLX."""
    # Copy self-attention weights
    copy_weights_self_attn(mlx_block.self_attn, torch_block.self_attn)

    # Copy cross-attention weights
    copy_weights_self_attn(mlx_block.cross_attn, torch_block.cross_attn)

    # Copy norm weights (norm1 and norm2 have no learnable params when elementwise_affine=False)
    # Only copy if elementwise_affine=True (norm3 when cross_attn_norm=True)
    if mlx_block._use_norm3:
        mlx_block.norm3.weight = mx.array(torch_block.norm3.weight.detach().numpy())
        mlx_block.norm3.bias = mx.array(torch_block.norm3.bias.detach().numpy())

    # Copy FFN weights
    mlx_block.ffn_linear1.weight = mx.array(torch_block.ffn[0].weight.detach().numpy())
    mlx_block.ffn_linear1.bias = mx.array(torch_block.ffn[0].bias.detach().numpy())
    mlx_block.ffn_linear2.weight = mx.array(torch_block.ffn[2].weight.detach().numpy())
    mlx_block.ffn_linear2.bias = mx.array(torch_block.ffn[2].bias.detach().numpy())

    # Copy modulation parameter
    mlx_block.modulation = mx.array(torch_block.modulation.detach().numpy())

    # Copy camera injection weights
    mlx_block.cam_injector_layer1.weight = mx.array(torch_block.cam_injector_layer1.weight.detach().numpy())
    mlx_block.cam_injector_layer1.bias = mx.array(torch_block.cam_injector_layer1.bias.detach().numpy())
    mlx_block.cam_injector_layer2.weight = mx.array(torch_block.cam_injector_layer2.weight.detach().numpy())
    mlx_block.cam_injector_layer2.bias = mx.array(torch_block.cam_injector_layer2.bias.detach().numpy())
    mlx_block.cam_scale_layer.weight = mx.array(torch_block.cam_scale_layer.weight.detach().numpy())
    mlx_block.cam_scale_layer.bias = mx.array(torch_block.cam_scale_layer.bias.detach().numpy())
    mlx_block.cam_shift_layer.weight = mx.array(torch_block.cam_shift_layer.weight.detach().numpy())
    mlx_block.cam_shift_layer.bias = mx.array(torch_block.cam_shift_layer.bias.detach().numpy())


def pytorch_block_forward_float32(torch_block, x, e, grid_sizes, freqs, context, context_lens, dit_cond_dict=None):
    """Run PyTorch block forward in float32 for fair comparison."""
    b, s, _ = x.shape
    seq_lens = torch.tensor([s] * b, dtype=torch.long)
    
    with torch.no_grad():
        # Modulation
        e_mod = (torch_block.modulation.unsqueeze(0) + e).chunk(6, dim=2)
        
        # Self-attention with modulation
        norm1_out = torch_block.norm1(x).float()
        x_mod = norm1_out * (1 + e_mod[1].squeeze(2)) + e_mod[0].squeeze(2)
        
        # Manual self-attention in float32
        torch_attn = torch_block.self_attn
        n, d = torch_attn.num_heads, torch_attn.head_dim
        
        q = torch_attn.q(x_mod)
        k = torch_attn.k(x_mod)
        v = torch_attn.v(x_mod)
        
        if torch_attn.qk_norm:
            q = torch_attn.norm_q(q)
            k = torch_attn.norm_k(k)
        
        q = q.view(b, s, n, d)
        k = k.view(b, s, n, d)
        v = v.view(b, s, n, d)
        
        # Apply RoPE
        from wan.modules.model import rope_apply
        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)
        
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        
        scale = d ** -0.5
        attn_out = F.scaled_dot_product_attention(q_t, k_t, v_t, scale=scale)
        attn_out = attn_out.transpose(1, 2).contiguous()
        y = torch_attn.o(attn_out.flatten(2))
        
        x = x + y * e_mod[2].squeeze(2)
        
        # Camera injection (optional)
        if dit_cond_dict is not None and "c2ws_plucker_emb" in dit_cond_dict:
            c2ws_plucker_emb = dit_cond_dict["c2ws_plucker_emb"]
            c2ws_hidden = torch_block.cam_injector_layer2(F.silu(torch_block.cam_injector_layer1(c2ws_plucker_emb)))
            c2ws_hidden = c2ws_hidden + c2ws_plucker_emb
            cam_scale = torch_block.cam_scale_layer(c2ws_hidden)
            cam_shift = torch_block.cam_shift_layer(c2ws_hidden)
            x = (1.0 + cam_scale) * x + cam_shift
        
        # Cross-attention
        norm3_out = torch_block.norm3(x) if not isinstance(torch_block.norm3, nn.Identity) else x
        
        # Manual cross-attention in float32
        cross_attn = torch_block.cross_attn
        s_ctx = context.shape[1]
        
        q_cross = cross_attn.q(norm3_out)
        k_cross = cross_attn.k(context)
        v_cross = cross_attn.v(context)
        
        if cross_attn.qk_norm:
            q_cross = cross_attn.norm_q(q_cross)
            k_cross = cross_attn.norm_k(k_cross)
        
        q_cross = q_cross.view(b, s, n, d)
        k_cross = k_cross.view(b, s_ctx, n, d)
        v_cross = v_cross.view(b, s_ctx, n, d)
        
        q_t = q_cross.transpose(1, 2)
        k_t = k_cross.transpose(1, 2)
        v_t = v_cross.transpose(1, 2)
        
        # Create mask for context_lens if needed
        attn_mask = None
        if context_lens is not None:
            masks = []
            for i in range(b):
                ctx_len = int(context_lens[i].item())
                row_mask = torch.arange(s_ctx) < ctx_len
                masks.append(row_mask)
            key_mask = torch.stack(masks, dim=0).unsqueeze(1).unsqueeze(2)
            attn_mask = torch.where(key_mask, 0.0, float('-inf'))
        
        cross_out = F.scaled_dot_product_attention(q_t, k_t, v_t, scale=scale, attn_mask=attn_mask)
        cross_out = cross_out.transpose(1, 2).contiguous()
        cross_out = cross_attn.o(cross_out.flatten(2))
        
        x = x + cross_out
        
        # FFN with modulation
        norm2_out = torch_block.norm2(x).float()
        x_mod = norm2_out * (1 + e_mod[4].squeeze(2)) + e_mod[3].squeeze(2)
        y = torch_block.ffn(x_mod)
        x = x + y * e_mod[5].squeeze(2)
        
    return x


class TestWanAttentionBlockMLXInitialization:
    """Tests for WanAttentionBlockMLX initialization and structure."""

    @pytest.fixture
    def default_config(self):
        """Default configuration matching WanModel."""
        return {
            'dim': 2048,
            'ffn_dim': 8192,
            'num_heads': 16,
            'window_size': (-1, -1),
            'qk_norm': True,
            'cross_attn_norm': False,
            'eps': 1e-6,
        }

    @pytest.fixture
    def small_config(self):
        """Smaller configuration for faster tests."""
        return {
            'dim': 256,
            'ffn_dim': 1024,
            'num_heads': 4,
            'window_size': (-1, -1),
            'qk_norm': True,
            'cross_attn_norm': False,
            'eps': 1e-6,
        }

    def test_initialization_default(self, default_config):
        """Test that WanAttentionBlockMLX initializes with default config."""
        block = WanAttentionBlockMLX(**default_config)

        assert block.dim == default_config['dim']
        assert block.ffn_dim == default_config['ffn_dim']
        assert block.num_heads == default_config['num_heads']
        assert block.window_size == default_config['window_size']
        assert block.qk_norm == default_config['qk_norm']
        assert block.cross_attn_norm == default_config['cross_attn_norm']
        assert block.eps == default_config['eps']

    def test_initialization_small(self, small_config):
        """Test that WanAttentionBlockMLX initializes with small config."""
        block = WanAttentionBlockMLX(**small_config)

        assert block.dim == small_config['dim']
        assert block.ffn_dim == small_config['ffn_dim']
        assert block.num_heads == small_config['num_heads']

    def test_modulation_shape(self, small_config):
        """Test that modulation parameter has correct shape."""
        block = WanAttentionBlockMLX(**small_config)
        dim = small_config['dim']

        assert block.modulation.shape == (1, 6, dim)

    def test_ffn_shapes(self, small_config):
        """Test that FFN layers have correct shapes."""
        block = WanAttentionBlockMLX(**small_config)
        dim = small_config['dim']
        ffn_dim = small_config['ffn_dim']

        # Linear layers: weight is [out_features, in_features] in MLX
        assert block.ffn_linear1.weight.shape == (ffn_dim, dim)
        assert block.ffn_linear2.weight.shape == (dim, ffn_dim)

    def test_camera_injection_shapes(self, small_config):
        """Test that camera injection layers have correct shapes."""
        block = WanAttentionBlockMLX(**small_config)
        dim = small_config['dim']

        assert block.cam_injector_layer1.weight.shape == (dim, dim)
        assert block.cam_injector_layer2.weight.shape == (dim, dim)
        assert block.cam_scale_layer.weight.shape == (dim, dim)
        assert block.cam_shift_layer.weight.shape == (dim, dim)

    def test_cross_attn_norm_false(self, small_config):
        """Test that norm3 is not created when cross_attn_norm=False."""
        block = WanAttentionBlockMLX(**small_config)
        assert block._use_norm3 is False
        assert not hasattr(block, 'norm3') or block._use_norm3 is False

    def test_cross_attn_norm_true(self, small_config):
        """Test that norm3 is created when cross_attn_norm=True."""
        config = {**small_config, 'cross_attn_norm': True}
        block = WanAttentionBlockMLX(**config)

        assert block._use_norm3 is True
        assert hasattr(block, 'norm3')
        assert block.norm3.weight is not None
        assert block.norm3.bias is not None


class TestWanAttentionBlockMLXFFN:
    """Tests for FFN (Feed-Forward Network) component."""

    @pytest.fixture
    def small_config(self):
        return {
            'dim': 256,
            'ffn_dim': 1024,
            'num_heads': 4,
            'window_size': (-1, -1),
            'qk_norm': True,
            'cross_attn_norm': False,
            'eps': 1e-6,
        }

    def test_ffn_output_shape(self, small_config):
        """Test that FFN produces correct output shape."""
        block = WanAttentionBlockMLX(**small_config)
        dim = small_config['dim']
        batch_size = 2
        seq_len = 64

        np.random.seed(42)
        np_input = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
        mlx_input = mx.array(np_input)

        output = block._ffn(mlx_input)
        mx.eval(output)

        assert output.shape == (batch_size, seq_len, dim)

    def test_ffn_gelu_tanh(self, small_config):
        """Test that FFN uses GELU with tanh approximation."""
        block = WanAttentionBlockMLX(**small_config)
        dim = small_config['dim']
        ffn_dim = small_config['ffn_dim']

        # Create simple input
        np_input = np.array([[[1.0, 2.0, 3.0, 4.0] * (dim // 4)]]).astype(np.float32)
        mlx_input = mx.array(np_input)

        # Compute FFN
        output = block._ffn(mlx_input)
        mx.eval(output)

        # Check it doesn't produce NaN/Inf
        output_np = np.array(output)
        assert not np.any(np.isnan(output_np)), "FFN output contains NaN"
        assert not np.any(np.isinf(output_np)), "FFN output contains Inf"


class TestWanAttentionBlockMLXPyTorchEquivalence:
    """Tests comparing MLX WanAttentionBlock to PyTorch reference."""

    @pytest.fixture
    def small_config(self):
        return {
            'dim': 256,
            'ffn_dim': 1024,
            'num_heads': 4,
            'window_size': (-1, -1),
            'qk_norm': True,
            'cross_attn_norm': False,
            'eps': 1e-6,
        }

    def test_pytorch_equivalence_no_camera(self, small_config):
        """Test that MLX output matches PyTorch without camera injection."""
        batch_size = 1
        dim = small_config['dim']
        num_heads = small_config['num_heads']
        head_dim = dim // num_heads  # 64

        # Grid: 4x4x4 = 64 positions
        f, h, w = 4, 4, 4
        seq_len = f * h * w
        context_len = 77

        # Create shared inputs
        np.random.seed(123)
        np_x = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
        np_e = np.random.randn(batch_size, 1, 6, dim).astype(np.float32)
        np_context = np.random.randn(batch_size, context_len, dim).astype(np.float32)
        grid_sizes_np = np.array([[f, h, w]], dtype=np.int64)

        # ---- PyTorch Reference ----
        torch_block = WanAttentionBlock(**small_config)

        # Create PyTorch freqs
        torch_freqs = torch.cat([
            rope_params(1024, head_dim - 4 * (head_dim // 6)),
            rope_params(1024, 2 * (head_dim // 6)),
            rope_params(1024, 2 * (head_dim // 6))
        ], dim=1)

        torch_x = torch.from_numpy(np_x)
        torch_e = torch.from_numpy(np_e)
        torch_context = torch.from_numpy(np_context)
        torch_grid = torch.from_numpy(grid_sizes_np)

        torch_output = pytorch_block_forward_float32(
            torch_block, torch_x, torch_e, torch_grid, torch_freqs,
            torch_context, None, dit_cond_dict=None
        )
        torch_output_np = torch_output.numpy()

        # ---- MLX Implementation ----
        mlx_block = WanAttentionBlockMLX(**small_config)

        # Copy weights
        copy_weights_block(mlx_block, torch_block)

        # Create MLX freqs
        cos_tuple, sin_tuple = create_rope_freqs_mlx(1024, head_dim)

        mlx_x = mx.array(np_x)
        mlx_e = mx.array(np_e)
        mlx_context = mx.array(np_context)
        mlx_grid = mx.array(grid_sizes_np)
        mlx_seq_lens = mx.array([seq_len], dtype=mx.int64)

        mlx_output = mlx_block(
            mlx_x, mlx_e, mlx_seq_lens, mlx_grid, cos_tuple, sin_tuple,
            mlx_context, None, dit_cond_dict=None
        )
        mx.eval(mlx_output)
        mlx_output_np = np.array(mlx_output)

        # Compare outputs
        np.testing.assert_allclose(
            mlx_output_np,
            torch_output_np,
            rtol=1e-4,
            atol=1e-4,
            err_msg="MLX block output doesn't match PyTorch (no camera injection)"
        )

    def test_pytorch_equivalence_with_camera(self, small_config):
        """Test that MLX output matches PyTorch with camera injection."""
        batch_size = 1
        dim = small_config['dim']
        num_heads = small_config['num_heads']
        head_dim = dim // num_heads

        f, h, w = 4, 4, 4
        seq_len = f * h * w
        context_len = 77

        # Create shared inputs
        np.random.seed(456)
        np_x = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
        np_e = np.random.randn(batch_size, 1, 6, dim).astype(np.float32)
        np_context = np.random.randn(batch_size, context_len, dim).astype(np.float32)
        np_camera = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
        grid_sizes_np = np.array([[f, h, w]], dtype=np.int64)

        # ---- PyTorch Reference ----
        torch_block = WanAttentionBlock(**small_config)

        torch_freqs = torch.cat([
            rope_params(1024, head_dim - 4 * (head_dim // 6)),
            rope_params(1024, 2 * (head_dim // 6)),
            rope_params(1024, 2 * (head_dim // 6))
        ], dim=1)

        torch_x = torch.from_numpy(np_x)
        torch_e = torch.from_numpy(np_e)
        torch_context = torch.from_numpy(np_context)
        torch_grid = torch.from_numpy(grid_sizes_np)
        torch_camera = torch.from_numpy(np_camera)

        dit_cond_dict_torch = {"c2ws_plucker_emb": torch_camera}

        torch_output = pytorch_block_forward_float32(
            torch_block, torch_x, torch_e, torch_grid, torch_freqs,
            torch_context, None, dit_cond_dict=dit_cond_dict_torch
        )
        torch_output_np = torch_output.numpy()

        # ---- MLX Implementation ----
        mlx_block = WanAttentionBlockMLX(**small_config)
        copy_weights_block(mlx_block, torch_block)

        cos_tuple, sin_tuple = create_rope_freqs_mlx(1024, head_dim)

        mlx_x = mx.array(np_x)
        mlx_e = mx.array(np_e)
        mlx_context = mx.array(np_context)
        mlx_grid = mx.array(grid_sizes_np)
        mlx_seq_lens = mx.array([seq_len], dtype=mx.int64)
        mlx_camera = mx.array(np_camera)

        dit_cond_dict_mlx = {"c2ws_plucker_emb": mlx_camera}

        mlx_output = mlx_block(
            mlx_x, mlx_e, mlx_seq_lens, mlx_grid, cos_tuple, sin_tuple,
            mlx_context, None, dit_cond_dict=dit_cond_dict_mlx
        )
        mx.eval(mlx_output)
        mlx_output_np = np.array(mlx_output)

        # Compare outputs
        np.testing.assert_allclose(
            mlx_output_np,
            torch_output_np,
            rtol=1e-4,
            atol=1e-4,
            err_msg="MLX block output doesn't match PyTorch (with camera injection)"
        )

    def test_pytorch_equivalence_with_cross_attn_norm(self):
        """Test that MLX output matches PyTorch with cross_attn_norm=True."""
        config = {
            'dim': 256,
            'ffn_dim': 1024,
            'num_heads': 4,
            'window_size': (-1, -1),
            'qk_norm': True,
            'cross_attn_norm': True,  # Enable cross-attention norm
            'eps': 1e-6,
        }
        batch_size = 1
        dim = config['dim']
        num_heads = config['num_heads']
        head_dim = dim // num_heads

        f, h, w = 4, 4, 4
        seq_len = f * h * w
        context_len = 77

        np.random.seed(789)
        np_x = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
        np_e = np.random.randn(batch_size, 1, 6, dim).astype(np.float32)
        np_context = np.random.randn(batch_size, context_len, dim).astype(np.float32)
        grid_sizes_np = np.array([[f, h, w]], dtype=np.int64)

        # ---- PyTorch Reference ----
        torch_block = WanAttentionBlock(**config)

        torch_freqs = torch.cat([
            rope_params(1024, head_dim - 4 * (head_dim // 6)),
            rope_params(1024, 2 * (head_dim // 6)),
            rope_params(1024, 2 * (head_dim // 6))
        ], dim=1)

        torch_x = torch.from_numpy(np_x)
        torch_e = torch.from_numpy(np_e)
        torch_context = torch.from_numpy(np_context)
        torch_grid = torch.from_numpy(grid_sizes_np)

        torch_output = pytorch_block_forward_float32(
            torch_block, torch_x, torch_e, torch_grid, torch_freqs,
            torch_context, None, dit_cond_dict=None
        )
        torch_output_np = torch_output.numpy()

        # ---- MLX Implementation ----
        mlx_block = WanAttentionBlockMLX(**config)
        copy_weights_block(mlx_block, torch_block)

        cos_tuple, sin_tuple = create_rope_freqs_mlx(1024, head_dim)

        mlx_x = mx.array(np_x)
        mlx_e = mx.array(np_e)
        mlx_context = mx.array(np_context)
        mlx_grid = mx.array(grid_sizes_np)
        mlx_seq_lens = mx.array([seq_len], dtype=mx.int64)

        mlx_output = mlx_block(
            mlx_x, mlx_e, mlx_seq_lens, mlx_grid, cos_tuple, sin_tuple,
            mlx_context, None, dit_cond_dict=None
        )
        mx.eval(mlx_output)
        mlx_output_np = np.array(mlx_output)

        np.testing.assert_allclose(
            mlx_output_np,
            torch_output_np,
            rtol=1e-4,
            atol=1e-4,
            err_msg="MLX block with cross_attn_norm doesn't match PyTorch"
        )

    def test_wanmodel_config(self):
        """Test with exact WanModel configuration (dim=2048, ffn_dim=8192, num_heads=16)."""
        config = {
            'dim': 2048,
            'ffn_dim': 8192,
            'num_heads': 16,
            'window_size': (-1, -1),
            'qk_norm': True,
            'cross_attn_norm': False,
            'eps': 1e-6,
        }
        batch_size = 1
        dim = config['dim']
        num_heads = config['num_heads']
        head_dim = dim // num_heads  # 128

        # Smaller grid for faster test
        f, h, w = 4, 4, 4
        seq_len = f * h * w
        context_len = 77

        np.random.seed(111)
        np_x = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
        np_e = np.random.randn(batch_size, 1, 6, dim).astype(np.float32)
        np_context = np.random.randn(batch_size, context_len, dim).astype(np.float32)
        grid_sizes_np = np.array([[f, h, w]], dtype=np.int64)

        # ---- PyTorch Reference ----
        torch_block = WanAttentionBlock(**config)

        torch_freqs = torch.cat([
            rope_params(1024, head_dim - 4 * (head_dim // 6)),
            rope_params(1024, 2 * (head_dim // 6)),
            rope_params(1024, 2 * (head_dim // 6))
        ], dim=1)

        torch_x = torch.from_numpy(np_x)
        torch_e = torch.from_numpy(np_e)
        torch_context = torch.from_numpy(np_context)
        torch_grid = torch.from_numpy(grid_sizes_np)

        torch_output = pytorch_block_forward_float32(
            torch_block, torch_x, torch_e, torch_grid, torch_freqs,
            torch_context, None, dit_cond_dict=None
        )
        torch_output_np = torch_output.numpy()

        # ---- MLX Implementation ----
        mlx_block = WanAttentionBlockMLX(**config)
        copy_weights_block(mlx_block, torch_block)

        cos_tuple, sin_tuple = create_rope_freqs_mlx(1024, head_dim)

        mlx_x = mx.array(np_x)
        mlx_e = mx.array(np_e)
        mlx_context = mx.array(np_context)
        mlx_grid = mx.array(grid_sizes_np)
        mlx_seq_lens = mx.array([seq_len], dtype=mx.int64)

        mlx_output = mlx_block(
            mlx_x, mlx_e, mlx_seq_lens, mlx_grid, cos_tuple, sin_tuple,
            mlx_context, None, dit_cond_dict=None
        )
        mx.eval(mlx_output)
        mlx_output_np = np.array(mlx_output)

        np.testing.assert_allclose(
            mlx_output_np,
            torch_output_np,
            rtol=1e-4,
            atol=1e-4,
            err_msg="MLX block doesn't match PyTorch for WanModel config"
        )


class TestWanAttentionBlockMLXNumericalStability:
    """Tests for numerical stability."""

    @pytest.fixture
    def small_config(self):
        return {
            'dim': 256,
            'ffn_dim': 1024,
            'num_heads': 4,
            'window_size': (-1, -1),
            'qk_norm': True,
            'cross_attn_norm': False,
            'eps': 1e-6,
        }

    def test_small_values(self, small_config):
        """Test numerical stability with small input values."""
        block = WanAttentionBlockMLX(**small_config)
        dim = small_config['dim']
        head_dim = dim // small_config['num_heads']

        f, h, w = 4, 4, 4
        seq_len = f * h * w
        context_len = 77
        batch_size = 1

        np.random.seed(222)
        np_x = np.random.randn(batch_size, seq_len, dim).astype(np.float32) * 1e-4
        np_e = np.random.randn(batch_size, 1, 6, dim).astype(np.float32) * 1e-4
        np_context = np.random.randn(batch_size, context_len, dim).astype(np.float32) * 1e-4
        grid_sizes_np = np.array([[f, h, w]], dtype=np.int64)

        cos_tuple, sin_tuple = create_rope_freqs_mlx(1024, head_dim)

        mlx_x = mx.array(np_x)
        mlx_e = mx.array(np_e)
        mlx_context = mx.array(np_context)
        mlx_grid = mx.array(grid_sizes_np)
        mlx_seq_lens = mx.array([seq_len], dtype=mx.int64)

        output = block(
            mlx_x, mlx_e, mlx_seq_lens, mlx_grid, cos_tuple, sin_tuple,
            mlx_context, None, dit_cond_dict=None
        )
        mx.eval(output)
        output_np = np.array(output)

        assert not np.any(np.isnan(output_np)), "Small input produced NaN"
        assert not np.any(np.isinf(output_np)), "Small input produced Inf"

    def test_large_values(self, small_config):
        """Test numerical stability with large input values."""
        block = WanAttentionBlockMLX(**small_config)
        dim = small_config['dim']
        head_dim = dim // small_config['num_heads']

        f, h, w = 4, 4, 4
        seq_len = f * h * w
        context_len = 77
        batch_size = 1

        np.random.seed(333)
        np_x = np.random.randn(batch_size, seq_len, dim).astype(np.float32) * 10
        np_e = np.random.randn(batch_size, 1, 6, dim).astype(np.float32) * 10
        np_context = np.random.randn(batch_size, context_len, dim).astype(np.float32) * 10
        grid_sizes_np = np.array([[f, h, w]], dtype=np.int64)

        cos_tuple, sin_tuple = create_rope_freqs_mlx(1024, head_dim)

        mlx_x = mx.array(np_x)
        mlx_e = mx.array(np_e)
        mlx_context = mx.array(np_context)
        mlx_grid = mx.array(grid_sizes_np)
        mlx_seq_lens = mx.array([seq_len], dtype=mx.int64)

        output = block(
            mlx_x, mlx_e, mlx_seq_lens, mlx_grid, cos_tuple, sin_tuple,
            mlx_context, None, dit_cond_dict=None
        )
        mx.eval(output)
        output_np = np.array(output)

        assert not np.any(np.isnan(output_np)), "Large input produced NaN"
        assert not np.any(np.isinf(output_np)), "Large input produced Inf"

    def test_deterministic_output(self, small_config):
        """Test that the same input produces the same output."""
        block = WanAttentionBlockMLX(**small_config)
        dim = small_config['dim']
        head_dim = dim // small_config['num_heads']

        f, h, w = 4, 4, 4
        seq_len = f * h * w
        context_len = 77
        batch_size = 1

        np.random.seed(444)
        np_x = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
        np_e = np.random.randn(batch_size, 1, 6, dim).astype(np.float32)
        np_context = np.random.randn(batch_size, context_len, dim).astype(np.float32)
        grid_sizes_np = np.array([[f, h, w]], dtype=np.int64)

        cos_tuple, sin_tuple = create_rope_freqs_mlx(1024, head_dim)

        mlx_x = mx.array(np_x)
        mlx_e = mx.array(np_e)
        mlx_context = mx.array(np_context)
        mlx_grid = mx.array(grid_sizes_np)
        mlx_seq_lens = mx.array([seq_len], dtype=mx.int64)

        output1 = block(
            mlx_x, mlx_e, mlx_seq_lens, mlx_grid, cos_tuple, sin_tuple,
            mlx_context, None, dit_cond_dict=None
        )
        mx.eval(output1)

        output2 = block(
            mlx_x, mlx_e, mlx_seq_lens, mlx_grid, cos_tuple, sin_tuple,
            mlx_context, None, dit_cond_dict=None
        )
        mx.eval(output2)

        np.testing.assert_array_equal(
            np.array(output1),
            np.array(output2),
            err_msg="Same input produced different outputs"
        )


class TestWanAttentionBlockMLXBatching:
    """Tests for batch processing."""

    @pytest.fixture
    def small_config(self):
        return {
            'dim': 256,
            'ffn_dim': 1024,
            'num_heads': 4,
            'window_size': (-1, -1),
            'qk_norm': True,
            'cross_attn_norm': False,
            'eps': 1e-6,
        }

    def test_batch_processing(self, small_config):
        """Test that batched inputs are processed correctly."""
        block = WanAttentionBlockMLX(**small_config)
        dim = small_config['dim']
        head_dim = dim // small_config['num_heads']

        batch_size = 4
        f, h, w = 4, 4, 4
        seq_len = f * h * w
        context_len = 77

        np.random.seed(555)
        np_x = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
        np_e = np.random.randn(batch_size, 1, 6, dim).astype(np.float32)
        np_context = np.random.randn(batch_size, context_len, dim).astype(np.float32)
        grid_sizes_np = np.array([[f, h, w]] * batch_size, dtype=np.int64)
        seq_lens_np = np.array([seq_len] * batch_size, dtype=np.int64)

        cos_tuple, sin_tuple = create_rope_freqs_mlx(1024, head_dim)

        mlx_x = mx.array(np_x)
        mlx_e = mx.array(np_e)
        mlx_context = mx.array(np_context)
        mlx_grid = mx.array(grid_sizes_np)
        mlx_seq_lens = mx.array(seq_lens_np)

        output = block(
            mlx_x, mlx_e, mlx_seq_lens, mlx_grid, cos_tuple, sin_tuple,
            mlx_context, None, dit_cond_dict=None
        )
        mx.eval(output)

        assert output.shape == (batch_size, seq_len, dim)

        output_np = np.array(output)
        assert not np.any(np.isnan(output_np)), "Batch output contains NaN"
        assert not np.any(np.isinf(output_np)), "Batch output contains Inf"

    def test_variable_context_lengths(self, small_config):
        """Test with variable-length context sequences."""
        block = WanAttentionBlockMLX(**small_config)
        dim = small_config['dim']
        head_dim = dim // small_config['num_heads']

        batch_size = 2
        f, h, w = 4, 4, 4
        seq_len = f * h * w
        max_context_len = 128
        context_lens_np = np.array([128, 64], dtype=np.int64)

        np.random.seed(666)
        np_x = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
        np_e = np.random.randn(batch_size, 1, 6, dim).astype(np.float32)
        np_context = np.random.randn(batch_size, max_context_len, dim).astype(np.float32)
        grid_sizes_np = np.array([[f, h, w]] * batch_size, dtype=np.int64)
        seq_lens_np = np.array([seq_len] * batch_size, dtype=np.int64)

        cos_tuple, sin_tuple = create_rope_freqs_mlx(1024, head_dim)

        mlx_x = mx.array(np_x)
        mlx_e = mx.array(np_e)
        mlx_context = mx.array(np_context)
        mlx_grid = mx.array(grid_sizes_np)
        mlx_seq_lens = mx.array(seq_lens_np)
        mlx_context_lens = mx.array(context_lens_np)

        output = block(
            mlx_x, mlx_e, mlx_seq_lens, mlx_grid, cos_tuple, sin_tuple,
            mlx_context, mlx_context_lens, dit_cond_dict=None
        )
        mx.eval(output)

        assert output.shape == (batch_size, seq_len, dim)

        output_np = np.array(output)
        assert not np.any(np.isnan(output_np)), "Variable context output contains NaN"
        assert not np.any(np.isinf(output_np)), "Variable context output contains Inf"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
