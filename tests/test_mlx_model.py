"""Unit tests for MLX WanModel implementation.

Tests that MLX implementation matches PyTorch reference within tolerance.

Key test areas:
- Initialization and shapes
- Patch embedding
- Text/time embeddings
- Forward pass PyTorch equivalence
- Weight loading from PyTorch checkpoints
- Unpatchify operation
"""

import math
import numpy as np
import pytest

# Import frameworks
import mlx.core as mx
import mlx.nn as nn_mlx
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import MLX implementations
from wan.backends.mlx.model import (
    WanModelMLX,
    PatchEmbedding3DMLX,
    sinusoidal_embedding_1d_mlx,
)
from wan.backends.mlx.rope import create_rope_freqs_mlx

# Import PyTorch reference
from wan.modules.model import WanModel, sinusoidal_embedding_1d, rope_params


class TestSinusoidalEmbedding:
    """Tests for sinusoidal position embeddings."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        dim = 256
        positions = mx.array([0, 1, 2, 3, 4])

        emb = sinusoidal_embedding_1d_mlx(dim, positions)
        mx.eval(emb)

        assert emb.shape == (5, dim)

    def test_pytorch_equivalence(self):
        """Test that MLX sinusoidal embeddings match PyTorch."""
        dim = 256
        positions = np.array([0, 1, 2, 3, 4, 10, 20, 50, 100], dtype=np.float32)

        # PyTorch
        torch_positions = torch.from_numpy(positions)
        torch_emb = sinusoidal_embedding_1d(dim, torch_positions)
        torch_emb_np = torch_emb.numpy()

        # MLX
        mlx_positions = mx.array(positions)
        mlx_emb = sinusoidal_embedding_1d_mlx(dim, mlx_positions)
        mx.eval(mlx_emb)
        mlx_emb_np = np.array(mlx_emb)

        np.testing.assert_allclose(
            mlx_emb_np, torch_emb_np, rtol=1e-5, atol=1e-5,
            err_msg="Sinusoidal embeddings don't match"
        )

    def test_dim_must_be_even(self):
        """Test that odd dimensions raise assertion."""
        with pytest.raises(AssertionError):
            sinusoidal_embedding_1d_mlx(255, mx.array([0, 1, 2]))


class TestPatchEmbedding3DMLX:
    """Tests for 3D patch embedding."""

    @pytest.fixture
    def default_config(self):
        return {
            'in_dim': 16,
            'out_dim': 2048,
            'patch_size': (1, 2, 2),
        }

    @pytest.fixture
    def small_config(self):
        return {
            'in_dim': 4,
            'out_dim': 256,
            'patch_size': (1, 2, 2),
        }

    def test_initialization(self, default_config):
        """Test that PatchEmbedding3DMLX initializes correctly."""
        embed = PatchEmbedding3DMLX(**default_config)

        expected_kernel = default_config['in_dim'] * math.prod(default_config['patch_size'])
        assert embed.weight.shape == (default_config['out_dim'], expected_kernel)
        assert embed.bias.shape == (default_config['out_dim'],)

    def test_output_shape(self, small_config):
        """Test that output shape is correct."""
        embed = PatchEmbedding3DMLX(**small_config)

        batch_size = 2
        in_dim = small_config['in_dim']
        f, h, w = 4, 8, 16
        pt, ph, pw = small_config['patch_size']

        np.random.seed(42)
        np_input = np.random.randn(batch_size, in_dim, f, h, w).astype(np.float32)
        mlx_input = mx.array(np_input)

        output = embed(mlx_input)
        mx.eval(output)

        expected_patches = (f // pt) * (h // ph) * (w // pw)
        assert output.shape == (batch_size, expected_patches, small_config['out_dim'])

    def test_unbatched_input(self, small_config):
        """Test that unbatched input is handled correctly."""
        embed = PatchEmbedding3DMLX(**small_config)

        in_dim = small_config['in_dim']
        f, h, w = 4, 8, 16
        pt, ph, pw = small_config['patch_size']

        np.random.seed(42)
        np_input = np.random.randn(in_dim, f, h, w).astype(np.float32)
        mlx_input = mx.array(np_input)

        output = embed(mlx_input)
        mx.eval(output)

        expected_patches = (f // pt) * (h // ph) * (w // pw)
        assert output.shape == (1, expected_patches, small_config['out_dim'])


class TestWanModelMLXInitialization:
    """Tests for WanModelMLX initialization."""

    @pytest.fixture
    def small_config(self):
        """Small configuration for faster tests."""
        return {
            'model_type': 'i2v',
            'patch_size': (1, 2, 2),
            'text_len': 128,
            'in_dim': 4,
            'dim': 256,
            'ffn_dim': 1024,
            'freq_dim': 64,
            'text_dim': 512,
            'out_dim': 4,
            'num_heads': 4,
            'num_layers': 2,
            'window_size': (-1, -1),
            'qk_norm': True,
            'cross_attn_norm': False,
            'eps': 1e-6,
        }

    @pytest.fixture
    def default_config(self):
        """Default WanModel configuration."""
        return {
            'model_type': 'i2v',
            'patch_size': (1, 2, 2),
            'text_len': 512,
            'in_dim': 16,
            'dim': 2048,
            'ffn_dim': 8192,
            'freq_dim': 256,
            'text_dim': 4096,
            'out_dim': 16,
            'num_heads': 16,
            'num_layers': 32,
            'window_size': (-1, -1),
            'qk_norm': True,
            'cross_attn_norm': True,
            'eps': 1e-6,
        }

    def test_initialization_small(self, small_config):
        """Test model initialization with small config."""
        model = WanModelMLX(**small_config)

        assert model.dim == small_config['dim']
        assert model.num_layers == small_config['num_layers']
        assert model.num_heads == small_config['num_heads']
        assert len(model.blocks) == small_config['num_layers']

    def test_initialization_default(self, default_config):
        """Test model initialization with default config."""
        model = WanModelMLX(**default_config)

        assert model.dim == 2048
        assert model.num_layers == 32
        assert model.num_heads == 16
        assert len(model.blocks) == 32

    def test_embedding_dimensions(self, small_config):
        """Test that embedding layers have correct dimensions."""
        model = WanModelMLX(**small_config)
        dim = small_config['dim']
        text_dim = small_config['text_dim']
        freq_dim = small_config['freq_dim']

        # Text embedding
        assert model.text_embedding_linear1.weight.shape == (dim, text_dim)
        assert model.text_embedding_linear2.weight.shape == (dim, dim)

        # Time embedding
        assert model.time_embedding_linear1.weight.shape == (dim, freq_dim)
        assert model.time_embedding_linear2.weight.shape == (dim, dim)

        # Time projection
        assert model.time_projection_linear.weight.shape == (dim * 6, dim)

    def test_rope_frequencies(self, small_config):
        """Test that RoPE frequencies are pre-computed."""
        model = WanModelMLX(**small_config)

        assert model._freqs_cos is not None
        assert model._freqs_sin is not None
        assert len(model._freqs_cos) == 3  # (frame, height, width)
        assert len(model._freqs_sin) == 3


class TestWanModelMLXForward:
    """Tests for WanModelMLX forward pass."""

    @pytest.fixture
    def small_config(self):
        # For i2v mode, in_dim must match the concatenated channel count
        # If x has 4 channels and y has 4 channels, concatenated = 8 channels
        return {
            'model_type': 'i2v',
            'patch_size': (1, 2, 2),
            'text_len': 64,
            'in_dim': 8,  # 4 from x + 4 from y (concatenated)
            'dim': 256,
            'ffn_dim': 1024,
            'freq_dim': 64,
            'text_dim': 512,
            'out_dim': 4,
            'num_heads': 4,
            'num_layers': 2,
            'window_size': (-1, -1),
            'qk_norm': True,
            'cross_attn_norm': False,
            'eps': 1e-6,
        }

    def test_forward_output_shape(self, small_config):
        """Test that forward pass produces correct output shape."""
        model = WanModelMLX(**small_config)

        # Each of x and y has in_dim/2 channels (4 each), concatenated to in_dim (8)
        x_channels = small_config['in_dim'] // 2
        out_dim = small_config['out_dim']
        f, h, w = 4, 8, 16

        np.random.seed(42)
        x = [mx.array(np.random.randn(x_channels, f, h, w).astype(np.float32))]
        y = [mx.array(np.random.randn(x_channels, f, h, w).astype(np.float32))]
        t = mx.array([0.5])
        context = [mx.array(np.random.randn(50, small_config['text_dim']).astype(np.float32))]

        seq_len = (f // small_config['patch_size'][0]) * \
                  (h // small_config['patch_size'][1]) * \
                  (w // small_config['patch_size'][2])

        output = model(x, t, context, seq_len, y=y)
        mx.eval(output[0])

        assert len(output) == 1
        # Output should be [out_dim, F, H, W] matching input spatial dims
        expected_out_c = out_dim
        expected_f = f
        expected_h = h
        expected_w = w
        assert output[0].shape == (expected_out_c, expected_f, expected_h, expected_w)

    def test_forward_t2v_mode(self):
        """Test forward pass in t2v mode (no y input)."""
        config = {
            'model_type': 't2v',
            'patch_size': (1, 2, 2),
            'text_len': 64,
            'in_dim': 4,
            'dim': 256,
            'ffn_dim': 1024,
            'freq_dim': 64,
            'text_dim': 512,
            'out_dim': 4,
            'num_heads': 4,
            'num_layers': 2,
        }
        model = WanModelMLX(**config)

        in_dim = config['in_dim']
        f, h, w = 4, 8, 16

        np.random.seed(42)
        x = [mx.array(np.random.randn(in_dim, f, h, w).astype(np.float32))]
        t = mx.array([0.5])
        context = [mx.array(np.random.randn(50, config['text_dim']).astype(np.float32))]

        seq_len = (f // config['patch_size'][0]) * \
                  (h // config['patch_size'][1]) * \
                  (w // config['patch_size'][2])

        output = model(x, t, context, seq_len, y=None)
        mx.eval(output[0])

        assert len(output) == 1
        assert output[0].shape == (config['out_dim'], f, h, w)

    def test_forward_with_camera(self, small_config):
        """Test forward pass with camera conditioning."""
        model = WanModelMLX(**small_config)

        # Each of x and y has in_dim/2 channels
        x_channels = small_config['in_dim'] // 2
        f, h, w = 4, 8, 16
        pt, ph, pw = small_config['patch_size']

        np.random.seed(42)
        x = [mx.array(np.random.randn(x_channels, f, h, w).astype(np.float32))]
        y = [mx.array(np.random.randn(x_channels, f, h, w).astype(np.float32))]
        t = mx.array([0.5])
        context = [mx.array(np.random.randn(50, small_config['text_dim']).astype(np.float32))]

        # Camera embedding: [1, 6*64, F, H, W]
        camera = mx.array(np.random.randn(1, 6 * 64, f, h, w).astype(np.float32))
        dit_cond_dict = {"c2ws_plucker_emb": [camera]}

        seq_len = (f // pt) * (h // ph) * (w // pw)

        output = model(x, t, context, seq_len, y=y, dit_cond_dict=dit_cond_dict)
        mx.eval(output[0])

        assert len(output) == 1
        output_np = np.array(output[0])
        assert not np.any(np.isnan(output_np)), "Camera conditioning produced NaN"

    def test_forward_numerical_stability(self, small_config):
        """Test numerical stability of forward pass."""
        model = WanModelMLX(**small_config)

        # Each of x and y has in_dim/2 channels
        x_channels = small_config['in_dim'] // 2
        f, h, w = 4, 8, 16

        np.random.seed(42)
        x = [mx.array(np.random.randn(x_channels, f, h, w).astype(np.float32))]
        y = [mx.array(np.random.randn(x_channels, f, h, w).astype(np.float32))]
        t = mx.array([0.5])
        context = [mx.array(np.random.randn(50, small_config['text_dim']).astype(np.float32))]

        seq_len = (f // small_config['patch_size'][0]) * \
                  (h // small_config['patch_size'][1]) * \
                  (w // small_config['patch_size'][2])

        output = model(x, t, context, seq_len, y=y)
        mx.eval(output[0])

        output_np = np.array(output[0])
        assert not np.any(np.isnan(output_np)), "Forward pass produced NaN"
        assert not np.any(np.isinf(output_np)), "Forward pass produced Inf"


class TestWanModelMLXUnpatchify:
    """Tests for unpatchify operation."""

    @pytest.fixture
    def small_config(self):
        return {
            'model_type': 'i2v',
            'patch_size': (1, 2, 2),
            'in_dim': 4,
            'dim': 256,
            'ffn_dim': 1024,
            'freq_dim': 64,
            'text_dim': 512,
            'out_dim': 4,
            'num_heads': 4,
            'num_layers': 2,
        }

    def test_unpatchify_shape(self, small_config):
        """Test that unpatchify produces correct output shape."""
        model = WanModelMLX(**small_config)

        f_out, h_out, w_out = 4, 8, 16
        pt, ph, pw = small_config['patch_size']
        out_dim = small_config['out_dim']
        num_patches = f_out * h_out * w_out
        patch_features = out_dim * pt * ph * pw

        np.random.seed(42)
        np_x = np.random.randn(1, num_patches + 10, patch_features).astype(np.float32)  # Extra padding
        mlx_x = mx.array(np_x)
        grid_sizes = mx.array([[f_out, h_out, w_out]], dtype=mx.int32)

        output = model._unpatchify(mlx_x, grid_sizes)
        mx.eval(output[0])

        expected_f = f_out * pt
        expected_h = h_out * ph
        expected_w = w_out * pw
        assert output[0].shape == (out_dim, expected_f, expected_h, expected_w)

    def test_unpatchify_batch(self, small_config):
        """Test unpatchify with batch of different grid sizes."""
        model = WanModelMLX(**small_config)

        pt, ph, pw = small_config['patch_size']
        out_dim = small_config['out_dim']
        patch_features = out_dim * pt * ph * pw

        # Two samples with different grid sizes
        f1, h1, w1 = 2, 4, 8
        f2, h2, w2 = 4, 4, 4
        max_patches = max(f1 * h1 * w1, f2 * h2 * w2)

        np.random.seed(42)
        np_x = np.random.randn(2, max_patches, patch_features).astype(np.float32)
        mlx_x = mx.array(np_x)
        grid_sizes = mx.array([[f1, h1, w1], [f2, h2, w2]], dtype=mx.int32)

        output = model._unpatchify(mlx_x, grid_sizes)
        for o in output:
            mx.eval(o)

        assert len(output) == 2
        assert output[0].shape == (out_dim, f1 * pt, h1 * ph, w1 * pw)
        assert output[1].shape == (out_dim, f2 * pt, h2 * ph, w2 * pw)


class TestWanModelMLXPyTorchEquivalence:
    """Tests comparing MLX WanModel to PyTorch reference."""

    @pytest.fixture
    def small_config(self):
        """Small shared config for equivalence tests."""
        return {
            'model_type': 'i2v',
            'patch_size': (1, 2, 2),
            'text_len': 64,
            'in_dim': 4,
            'dim': 256,
            'ffn_dim': 1024,
            'freq_dim': 64,
            'text_dim': 512,
            'out_dim': 4,
            'num_heads': 4,
            'num_layers': 2,
            'window_size': (-1, -1),
            'qk_norm': True,
            'cross_attn_norm': False,
            'eps': 1e-6,
        }

    def _copy_pytorch_weights_to_mlx(self, mlx_model, torch_model):
        """Copy all weights from PyTorch model to MLX model."""
        state_dict = torch_model.state_dict()

        for name, param in state_dict.items():
            np_param = param.detach().numpy()
            mlx_param = mx.array(np_param)

            # Handle weight transposition for Linear layers
            parts = name.split('.')

            if name.startswith('patch_embedding.'):
                if name == 'patch_embedding.weight':
                    # Conv3d weight: flatten
                    w = mlx_param.reshape(mlx_param.shape[0], -1)
                    mlx_model.patch_embedding.weight = w
                elif name == 'patch_embedding.bias':
                    mlx_model.patch_embedding.bias = mlx_param

            elif name.startswith('patch_embedding_wancamctrl.'):
                attr = name.split('.')[-1]
                if attr == 'weight':
                    mlx_param = mlx_param.T  # Transpose Linear weight
                setattr(mlx_model.patch_embedding_wancamctrl, attr, mlx_param)

            elif name.startswith('c2ws_hidden_states_layer1.'):
                attr = name.split('.')[-1]
                if attr == 'weight':
                    mlx_param = mlx_param.T
                setattr(mlx_model.c2ws_hidden_states_layer1, attr, mlx_param)

            elif name.startswith('c2ws_hidden_states_layer2.'):
                attr = name.split('.')[-1]
                if attr == 'weight':
                    mlx_param = mlx_param.T
                setattr(mlx_model.c2ws_hidden_states_layer2, attr, mlx_param)

            elif name.startswith('text_embedding.'):
                idx = int(parts[1])
                attr = parts[2]
                if attr == 'weight':
                    mlx_param = mlx_param.T
                if idx == 0:
                    setattr(mlx_model.text_embedding_linear1, attr, mlx_param)
                elif idx == 2:
                    setattr(mlx_model.text_embedding_linear2, attr, mlx_param)

            elif name.startswith('time_embedding.'):
                idx = int(parts[1])
                attr = parts[2]
                if attr == 'weight':
                    mlx_param = mlx_param.T
                if idx == 0:
                    setattr(mlx_model.time_embedding_linear1, attr, mlx_param)
                elif idx == 2:
                    setattr(mlx_model.time_embedding_linear2, attr, mlx_param)

            elif name.startswith('time_projection.'):
                attr = parts[2]
                if attr == 'weight':
                    mlx_param = mlx_param.T
                setattr(mlx_model.time_projection_linear, attr, mlx_param)

            elif name.startswith('blocks.'):
                self._copy_block_weight(mlx_model, name, mlx_param, np_param)

            elif name.startswith('head.'):
                self._copy_head_weight(mlx_model, name, mlx_param, np_param)

    def _copy_block_weight(self, mlx_model, name, mlx_param, np_param):
        """Copy a single weight into a block."""
        parts = name.split('.')
        block_idx = int(parts[1])
        if block_idx >= len(mlx_model.blocks):
            return

        block = mlx_model.blocks[block_idx]
        remainder = '.'.join(parts[2:])

        # Determine if transpose is needed (for Linear weights)
        needs_transpose = (
            'weight' in name and
            len(np_param.shape) == 2 and
            any(x in name for x in ['q.', 'k.', 'v.', 'o.', 'ffn.', 'cam_', 'norm_'])
        )

        if needs_transpose and not any(x in name for x in ['norm_q', 'norm_k', 'norm3']):
            mlx_param = mlx_param.T

        if remainder.startswith('self_attn.'):
            sub_parts = remainder.split('.')
            module = block.self_attn
            for p in sub_parts[1:-1]:
                module = getattr(module, p)
            setattr(module, sub_parts[-1], mlx_param)

        elif remainder.startswith('cross_attn.'):
            sub_parts = remainder.split('.')
            module = block.cross_attn
            for p in sub_parts[1:-1]:
                module = getattr(module, p)
            setattr(module, sub_parts[-1], mlx_param)

        elif remainder.startswith('ffn.'):
            sub_parts = remainder.split('.')
            idx = int(sub_parts[1])
            attr = sub_parts[2]
            if idx == 0:
                setattr(block.ffn_linear1, attr, mlx_param)
            elif idx == 2:
                setattr(block.ffn_linear2, attr, mlx_param)

        elif remainder == 'modulation':
            block.modulation = mlx_param

        elif remainder.startswith('cam_injector_layer1.'):
            attr = remainder.split('.')[-1]
            setattr(block.cam_injector_layer1, attr, mlx_param)

        elif remainder.startswith('cam_injector_layer2.'):
            attr = remainder.split('.')[-1]
            setattr(block.cam_injector_layer2, attr, mlx_param)

        elif remainder.startswith('cam_scale_layer.'):
            attr = remainder.split('.')[-1]
            setattr(block.cam_scale_layer, attr, mlx_param)

        elif remainder.startswith('cam_shift_layer.'):
            attr = remainder.split('.')[-1]
            setattr(block.cam_shift_layer, attr, mlx_param)

    def _copy_head_weight(self, mlx_model, name, mlx_param, np_param):
        """Copy a single weight into the head."""
        parts = name.split('.')
        remainder = '.'.join(parts[1:])

        if remainder.startswith('head.'):
            attr = remainder.split('.')[-1]
            if attr == 'weight' and len(np_param.shape) == 2:
                mlx_param = mlx_param.T
            setattr(mlx_model.head.head, attr, mlx_param)

        elif remainder == 'modulation':
            mlx_model.head.modulation = mlx_param

    def test_text_embedding_equivalence(self, small_config):
        """Test that text embedding matches PyTorch."""
        np.random.seed(42)
        np_input = np.random.randn(1, 50, small_config['text_dim']).astype(np.float32)

        # PyTorch
        torch_model = WanModel(**small_config)
        torch_input = torch.from_numpy(np_input)
        with torch.no_grad():
            torch_output = torch_model.text_embedding(torch_input)
        torch_output_np = torch_output.numpy()

        # MLX
        mlx_model = WanModelMLX(**small_config)

        # Copy weights
        # MLX nn.Linear stores weights as [out_features, in_features] same as PyTorch
        # So we DON'T transpose
        for idx, torch_layer in enumerate(torch_model.text_embedding):
            if isinstance(torch_layer, nn.Linear):
                w = torch_layer.weight.detach().numpy()  # No transpose needed
                b = torch_layer.bias.detach().numpy()
                if idx == 0:
                    mlx_model.text_embedding_linear1.weight = mx.array(w)
                    mlx_model.text_embedding_linear1.bias = mx.array(b)
                elif idx == 2:
                    mlx_model.text_embedding_linear2.weight = mx.array(w)
                    mlx_model.text_embedding_linear2.bias = mx.array(b)

        mlx_input = mx.array(np_input)
        mlx_output = mlx_model._text_embedding(mlx_input)
        mx.eval(mlx_output)
        mlx_output_np = np.array(mlx_output)

        np.testing.assert_allclose(
            mlx_output_np, torch_output_np, rtol=1e-4, atol=1e-4,
            err_msg="Text embedding doesn't match"
        )

    def test_time_embedding_equivalence(self, small_config):
        """Test that time embedding matches PyTorch."""
        np.random.seed(42)
        np_input = np.random.randn(1, 64, small_config['freq_dim']).astype(np.float32)

        # PyTorch
        torch_model = WanModel(**small_config)
        torch_input = torch.from_numpy(np_input)
        with torch.no_grad():
            torch_output = torch_model.time_embedding(torch_input)
        torch_output_np = torch_output.numpy()

        # MLX
        mlx_model = WanModelMLX(**small_config)

        # Copy weights
        # MLX nn.Linear stores weights as [out_features, in_features] same as PyTorch
        # So we DON'T transpose
        for idx, torch_layer in enumerate(torch_model.time_embedding):
            if isinstance(torch_layer, nn.Linear):
                w = torch_layer.weight.detach().numpy()  # No transpose needed
                b = torch_layer.bias.detach().numpy()
                if idx == 0:
                    mlx_model.time_embedding_linear1.weight = mx.array(w)
                    mlx_model.time_embedding_linear1.bias = mx.array(b)
                elif idx == 2:
                    mlx_model.time_embedding_linear2.weight = mx.array(w)
                    mlx_model.time_embedding_linear2.bias = mx.array(b)

        mlx_input = mx.array(np_input)
        mlx_output = mlx_model._time_embedding(mlx_input)
        mx.eval(mlx_output)
        mlx_output_np = np.array(mlx_output)

        np.testing.assert_allclose(
            mlx_output_np, torch_output_np, rtol=1e-4, atol=1e-4,
            err_msg="Time embedding doesn't match"
        )


class TestWanModelMLXConfigInference:
    """Tests for configuration inference from weights."""

    def test_infer_dim(self):
        """Test dimension inference from weights."""
        # After converter: Linear weight is [in_features, out_features]
        # For q projection: in=dim=2048, out=dim=2048
        # For ffn.0: in=dim=2048, out=ffn_dim=8192
        weights = {
            'blocks.0.self_attn.q.weight': mx.zeros((2048, 2048)),
            'blocks.0.ffn.0.weight': mx.zeros((2048, 8192)),  # [in=dim, out=ffn_dim]
        }

        config = WanModelMLX._infer_config_from_weights(weights)

        assert config.get('dim') == 2048
        assert config.get('ffn_dim') == 8192

    def test_infer_num_layers(self):
        """Test layer count inference from weights."""
        weights = {
            'blocks.0.self_attn.q.weight': mx.zeros((256, 256)),
            'blocks.1.self_attn.q.weight': mx.zeros((256, 256)),
            'blocks.2.self_attn.q.weight': mx.zeros((256, 256)),
            'blocks.3.self_attn.q.weight': mx.zeros((256, 256)),
        }

        config = WanModelMLX._infer_config_from_weights(weights)

        assert config.get('num_layers') == 4


class TestWanModelMLXNumericalStability:
    """Tests for numerical stability."""

    @pytest.fixture
    def small_config(self):
        # For i2v mode, in_dim must match the concatenated channel count
        return {
            'model_type': 'i2v',
            'patch_size': (1, 2, 2),
            'text_len': 64,
            'in_dim': 8,  # 4 from x + 4 from y (concatenated)
            'dim': 256,
            'ffn_dim': 1024,
            'freq_dim': 64,
            'text_dim': 512,
            'out_dim': 4,
            'num_heads': 4,
            'num_layers': 2,
        }

    def test_small_timesteps(self, small_config):
        """Test with very small timesteps."""
        model = WanModelMLX(**small_config)

        x_channels = small_config['in_dim'] // 2
        f, h, w = 4, 8, 16

        np.random.seed(42)
        x = [mx.array(np.random.randn(x_channels, f, h, w).astype(np.float32))]
        y = [mx.array(np.random.randn(x_channels, f, h, w).astype(np.float32))]
        t = mx.array([1e-6])  # Very small timestep
        context = [mx.array(np.random.randn(50, small_config['text_dim']).astype(np.float32))]

        seq_len = (f // small_config['patch_size'][0]) * \
                  (h // small_config['patch_size'][1]) * \
                  (w // small_config['patch_size'][2])

        output = model(x, t, context, seq_len, y=y)
        mx.eval(output[0])

        output_np = np.array(output[0])
        assert not np.any(np.isnan(output_np)), "Small timestep produced NaN"

    def test_large_timesteps(self, small_config):
        """Test with large timesteps."""
        model = WanModelMLX(**small_config)

        x_channels = small_config['in_dim'] // 2
        f, h, w = 4, 8, 16

        np.random.seed(42)
        x = [mx.array(np.random.randn(x_channels, f, h, w).astype(np.float32))]
        y = [mx.array(np.random.randn(x_channels, f, h, w).astype(np.float32))]
        t = mx.array([999.0])  # Large timestep
        context = [mx.array(np.random.randn(50, small_config['text_dim']).astype(np.float32))]

        seq_len = (f // small_config['patch_size'][0]) * \
                  (h // small_config['patch_size'][1]) * \
                  (w // small_config['patch_size'][2])

        output = model(x, t, context, seq_len, y=y)
        mx.eval(output[0])

        output_np = np.array(output[0])
        assert not np.any(np.isnan(output_np)), "Large timestep produced NaN"

    def test_deterministic_output(self, small_config):
        """Test that same input produces same output."""
        model = WanModelMLX(**small_config)

        x_channels = small_config['in_dim'] // 2
        f, h, w = 4, 8, 16

        np.random.seed(42)
        np_x = np.random.randn(x_channels, f, h, w).astype(np.float32)
        np_y = np.random.randn(x_channels, f, h, w).astype(np.float32)
        np_context = np.random.randn(50, small_config['text_dim']).astype(np.float32)

        seq_len = (f // small_config['patch_size'][0]) * \
                  (h // small_config['patch_size'][1]) * \
                  (w // small_config['patch_size'][2])

        # First run
        x1 = [mx.array(np_x)]
        y1 = [mx.array(np_y)]
        t1 = mx.array([0.5])
        context1 = [mx.array(np_context)]
        output1 = model(x1, t1, context1, seq_len, y=y1)
        mx.eval(output1[0])
        out1_np = np.array(output1[0])

        # Second run
        x2 = [mx.array(np_x)]
        y2 = [mx.array(np_y)]
        t2 = mx.array([0.5])
        context2 = [mx.array(np_context)]
        output2 = model(x2, t2, context2, seq_len, y=y2)
        mx.eval(output2[0])
        out2_np = np.array(output2[0])

        np.testing.assert_array_equal(
            out1_np, out2_np,
            err_msg="Same input produced different outputs"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
