"""Unit tests for MLX weight converter.

Tests that weight conversion correctly handles:
- Linear weight transposition (PyTorch [out, in] -> MLX [in, out])
- safetensors and .pth format loading
- Sharded checkpoint loading
- Caching with version hash
- Shape validation
- Lazy loading for memory efficiency
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, patch

import mlx.core as mx
import numpy as np
import pytest
import torch
import torch.nn as nn

from wan.backends.mlx.convert import (
    ShardedCheckpointLoader,
    WeightConverter,
    compute_checkpoint_hash,
    convert_pytorch_to_mlx,
    convert_weight,
    get_cache_path,
    is_linear_weight,
    load_mlx_weights,
    needs_transpose,
    validate_weight_shapes,
)


class TestIsLinearWeight:
    """Tests for is_linear_weight function."""

    def test_attention_q_weight(self):
        """Q projection weight should be identified as linear."""
        assert is_linear_weight("blocks.0.self_attn.q.weight")
        assert is_linear_weight("blocks.31.cross_attn.q.weight")

    def test_attention_k_weight(self):
        """K projection weight should be identified as linear."""
        assert is_linear_weight("blocks.0.self_attn.k.weight")
        assert is_linear_weight("blocks.31.cross_attn.k.weight")

    def test_attention_v_weight(self):
        """V projection weight should be identified as linear."""
        assert is_linear_weight("blocks.0.self_attn.v.weight")
        assert is_linear_weight("blocks.31.cross_attn.v.weight")

    def test_attention_o_weight(self):
        """Output projection weight should be identified as linear."""
        assert is_linear_weight("blocks.0.self_attn.o.weight")
        assert is_linear_weight("blocks.31.cross_attn.o.weight")

    def test_ffn_weights(self):
        """FFN linear weights should be identified."""
        assert is_linear_weight("blocks.0.ffn.0.weight")
        assert is_linear_weight("blocks.0.ffn.2.weight")

    def test_embedding_weights(self):
        """Embedding layer weights should be identified as linear."""
        assert is_linear_weight("text_embedding.0.weight")
        assert is_linear_weight("text_embedding.2.weight")
        assert is_linear_weight("time_embedding.0.weight")
        assert is_linear_weight("time_embedding.2.weight")
        assert is_linear_weight("time_projection.1.weight")

    def test_cam_control_weights(self):
        """Camera control layer weights should be identified."""
        assert is_linear_weight("blocks.0.cam_injector_layer1.weight")
        assert is_linear_weight("blocks.0.cam_injector_layer2.weight")
        assert is_linear_weight("blocks.0.cam_scale_layer.weight")
        assert is_linear_weight("blocks.0.cam_shift_layer.weight")
        assert is_linear_weight("c2ws_hidden_states_layer1.weight")
        assert is_linear_weight("c2ws_hidden_states_layer2.weight")
        assert is_linear_weight("patch_embedding_wancamctrl.weight")

    def test_head_weight(self):
        """Head linear layer weight should be identified."""
        assert is_linear_weight("head.head.weight")

    def test_bias_not_linear(self):
        """Bias terms should NOT be identified as linear weights."""
        assert not is_linear_weight("blocks.0.self_attn.q.bias")
        assert not is_linear_weight("blocks.0.ffn.0.bias")
        assert not is_linear_weight("text_embedding.0.bias")

    def test_norm_weights_not_linear(self):
        """Normalization layer weights should NOT need transpose."""
        assert not is_linear_weight("blocks.0.self_attn.norm_q.weight")
        assert not is_linear_weight("blocks.0.self_attn.norm_k.weight")
        assert not is_linear_weight("blocks.0.norm3.weight")

    def test_modulation_not_linear(self):
        """Modulation parameters should NOT be identified as linear."""
        assert not is_linear_weight("blocks.0.modulation")
        assert not is_linear_weight("head.modulation")

    def test_conv3d_weight_not_linear(self):
        """Conv3d weight (patch_embedding) should NOT be identified as linear."""
        assert not is_linear_weight("patch_embedding.weight")


class TestNeedsTranspose:
    """Tests for needs_transpose function."""

    def test_2d_linear_weight_needs_transpose(self):
        """2D linear weights should need transpose."""
        assert needs_transpose("blocks.0.self_attn.q.weight", (2048, 2048))
        assert needs_transpose("blocks.0.ffn.0.weight", (8192, 2048))

    def test_1d_weight_no_transpose(self):
        """1D weights (bias, norm) should NOT need transpose."""
        assert not needs_transpose("blocks.0.self_attn.q.bias", (2048,))
        assert not needs_transpose("blocks.0.self_attn.norm_q.weight", (2048,))

    def test_3d_weight_no_transpose(self):
        """3D+ weights (modulation) should NOT need transpose."""
        assert not needs_transpose("blocks.0.modulation", (1, 6, 2048))

    def test_5d_conv_weight_no_transpose(self):
        """5D Conv3d weights should NOT need transpose."""
        assert not needs_transpose("patch_embedding.weight", (2048, 16, 1, 2, 2))


class TestConvertWeight:
    """Tests for convert_weight function."""

    def test_linear_weight_transpose(self):
        """Linear weight should be transposed."""
        np_weight = np.random.randn(2048, 1024).astype(np.float32)
        mlx_weight = convert_weight(np_weight, "blocks.0.self_attn.q.weight")

        assert mlx_weight.shape == (1024, 2048)
        mx.eval(mlx_weight)

    def test_bias_no_transpose(self):
        """Bias should NOT be transposed."""
        np_bias = np.random.randn(2048).astype(np.float32)
        mlx_bias = convert_weight(np_bias, "blocks.0.self_attn.q.bias")

        assert mlx_bias.shape == (2048,)
        mx.eval(mlx_bias)

    def test_norm_weight_no_transpose(self):
        """Norm weight should NOT be transposed."""
        np_weight = np.random.randn(2048).astype(np.float32)
        mlx_weight = convert_weight(np_weight, "blocks.0.self_attn.norm_q.weight")

        assert mlx_weight.shape == (2048,)
        mx.eval(mlx_weight)

    def test_dtype_conversion_float16_to_float32(self):
        """Float16 should be converted to float32."""
        np_weight = np.random.randn(256, 128).astype(np.float16)
        mlx_weight = convert_weight(np_weight, "blocks.0.self_attn.q.weight")

        assert mlx_weight.dtype == mx.float32
        mx.eval(mlx_weight)

    def test_values_preserved_after_transpose(self):
        """Values should be correctly transposed."""
        np_weight = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        mlx_weight = convert_weight(np_weight, "blocks.0.self_attn.q.weight")
        mx.eval(mlx_weight)

        expected = np.array([[1, 4], [2, 5], [3, 6]], dtype=np.float32)
        np.testing.assert_array_equal(np.array(mlx_weight), expected)


class TestWeightConverter:
    """Tests for WeightConverter class."""

    @pytest.fixture
    def converter(self, tmp_path):
        """Create a converter with temporary cache directory."""
        return WeightConverter(cache_dir=tmp_path)

    @pytest.fixture
    def sample_state_dict(self):
        """Create a sample state dict with typical WanModel parameters."""
        return {
            "blocks.0.self_attn.q.weight": np.random.randn(2048, 2048).astype(
                np.float32
            ),
            "blocks.0.self_attn.q.bias": np.random.randn(2048).astype(np.float32),
            "blocks.0.self_attn.norm_q.weight": np.random.randn(2048).astype(
                np.float32
            ),
            "blocks.0.modulation": np.random.randn(1, 6, 2048).astype(np.float32),
            "blocks.0.ffn.0.weight": np.random.randn(8192, 2048).astype(np.float32),
            "blocks.0.ffn.0.bias": np.random.randn(8192).astype(np.float32),
        }

    def test_convert_state_dict_shapes(self, converter, sample_state_dict):
        """Test that converted state dict has correct shapes."""
        mlx_state_dict = converter.convert_state_dict(sample_state_dict)

        # Linear weight should be transposed
        assert mlx_state_dict["blocks.0.self_attn.q.weight"].shape == (2048, 2048)
        assert mlx_state_dict["blocks.0.ffn.0.weight"].shape == (2048, 8192)

        # Bias should keep original shape
        assert mlx_state_dict["blocks.0.self_attn.q.bias"].shape == (2048,)
        assert mlx_state_dict["blocks.0.ffn.0.bias"].shape == (8192,)

        # Norm weight should keep original shape
        assert mlx_state_dict["blocks.0.self_attn.norm_q.weight"].shape == (2048,)

        # Modulation should keep original shape
        assert mlx_state_dict["blocks.0.modulation"].shape == (1, 6, 2048)

    def test_convert_state_dict_types(self, converter, sample_state_dict):
        """Test that all converted weights are MLX arrays."""
        mlx_state_dict = converter.convert_state_dict(sample_state_dict)

        for name, weight in mlx_state_dict.items():
            assert isinstance(weight, mx.array), f"{name} is not mx.array"
            mx.eval(weight)

    def test_convert_state_dict_no_nan(self, converter, sample_state_dict):
        """Test that conversion doesn't produce NaN."""
        mlx_state_dict = converter.convert_state_dict(sample_state_dict)

        for name, weight in mlx_state_dict.items():
            mx.eval(weight)
            np_weight = np.array(weight)
            assert not np.any(np.isnan(np_weight)), f"{name} contains NaN"


class TestCaching:
    """Tests for weight caching functionality."""

    def test_get_cache_path_includes_hash(self, tmp_path):
        """Cache path should include content hash."""
        # Create a test file
        test_file = tmp_path / "test.safetensors"
        test_file.write_bytes(b"test content")

        cache_path = get_cache_path(test_file, cache_dir=tmp_path)

        assert "test_" in cache_path.name
        assert cache_path.suffix == ".safetensors"

    def test_hash_changes_with_content(self, tmp_path):
        """Hash should change when file size changes."""
        test_file = tmp_path / "test.safetensors"

        # Hash is based on name + size + mtime, so use different sizes
        test_file.write_bytes(b"content v1")
        hash1 = compute_checkpoint_hash(test_file)

        test_file.write_bytes(b"content v2 with much more data to change size")
        hash2 = compute_checkpoint_hash(test_file)

        assert hash1 != hash2

    def test_hash_for_sharded_checkpoint(self, tmp_path):
        """Hash should work for sharded checkpoints with index file."""
        # Create mock sharded checkpoint
        index = {
            "metadata": {"total_size": 1000},
            "weight_map": {"param1": "shard1.safetensors"},
        }
        index_file = tmp_path / "model.safetensors.index.json"
        index_file.write_text(json.dumps(index))

        ckpt_hash = compute_checkpoint_hash(tmp_path)
        assert len(ckpt_hash) == 16


class TestSafetensorsLoading:
    """Tests for safetensors format loading."""

    @pytest.fixture
    def sample_safetensors(self, tmp_path):
        """Create a sample safetensors file."""
        from safetensors.numpy import save_file

        weights = {
            "linear.weight": np.random.randn(256, 128).astype(np.float32),
            "linear.bias": np.random.randn(256).astype(np.float32),
        }

        path = tmp_path / "model.safetensors"
        save_file(weights, str(path))
        return path

    def test_load_safetensors_file(self, sample_safetensors, tmp_path):
        """Test loading a single safetensors file."""
        converter = WeightConverter(cache_dir=tmp_path, use_cache=False)
        state_dict = converter.load_checkpoint(sample_safetensors)

        assert "linear.weight" in state_dict
        assert "linear.bias" in state_dict
        assert state_dict["linear.weight"].shape == (256, 128)


class TestPthLoading:
    """Tests for PyTorch .pth format loading."""

    @pytest.fixture
    def sample_pth(self, tmp_path):
        """Create a sample .pth file."""
        model = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
        )

        path = tmp_path / "model.pth"
        torch.save(model.state_dict(), path)
        return path

    def test_load_pth_file(self, sample_pth, tmp_path):
        """Test loading a .pth file."""
        converter = WeightConverter(cache_dir=tmp_path, use_cache=False)
        state_dict = converter.load_checkpoint(sample_pth)

        assert "0.weight" in state_dict
        assert "0.bias" in state_dict
        assert "2.weight" in state_dict
        assert state_dict["0.weight"].shape == (256, 128)

    @pytest.fixture
    def nested_pth(self, tmp_path):
        """Create a .pth file with nested state_dict."""
        model = nn.Linear(64, 32)

        path = tmp_path / "checkpoint.pth"
        torch.save({"state_dict": model.state_dict(), "epoch": 10}, path)
        return path

    def test_load_nested_pth(self, nested_pth, tmp_path):
        """Test loading a .pth file with nested state_dict."""
        converter = WeightConverter(cache_dir=tmp_path, use_cache=False)
        state_dict = converter.load_checkpoint(nested_pth)

        assert "weight" in state_dict
        assert "bias" in state_dict


class TestShardedCheckpointLoader:
    """Tests for ShardedCheckpointLoader class."""

    @pytest.fixture
    def sharded_checkpoint(self, tmp_path):
        """Create a mock sharded checkpoint."""
        from safetensors.numpy import save_file

        # Create shard files
        shard1_weights = {
            "blocks.0.q.weight": np.random.randn(64, 64).astype(np.float32),
            "blocks.0.q.bias": np.random.randn(64).astype(np.float32),
        }
        shard2_weights = {
            "blocks.1.q.weight": np.random.randn(64, 64).astype(np.float32),
            "blocks.1.q.bias": np.random.randn(64).astype(np.float32),
        }

        save_file(shard1_weights, str(tmp_path / "model-00001-of-00002.safetensors"))
        save_file(shard2_weights, str(tmp_path / "model-00002-of-00002.safetensors"))

        # Create index file
        index = {
            "metadata": {"total_size": 1000},
            "weight_map": {
                "blocks.0.q.weight": "model-00001-of-00002.safetensors",
                "blocks.0.q.bias": "model-00001-of-00002.safetensors",
                "blocks.1.q.weight": "model-00002-of-00002.safetensors",
                "blocks.1.q.bias": "model-00002-of-00002.safetensors",
            },
        }

        index_file = tmp_path / "model.safetensors.index.json"
        index_file.write_text(json.dumps(index))

        return tmp_path

    def test_loader_initialization(self, sharded_checkpoint):
        """Test that loader initializes correctly."""
        loader = ShardedCheckpointLoader(sharded_checkpoint)

        assert len(loader.parameter_names) == 4
        assert len(loader.shard_files) == 2

    def test_get_weight(self, sharded_checkpoint):
        """Test getting individual weights."""
        loader = ShardedCheckpointLoader(sharded_checkpoint)

        weight = loader.get_weight("blocks.0.q.weight")
        assert weight.shape == (64, 64)

        bias = loader.get_weight("blocks.0.q.bias")
        assert bias.shape == (64,)

    def test_get_weight_missing(self, sharded_checkpoint):
        """Test that missing weight raises KeyError."""
        loader = ShardedCheckpointLoader(sharded_checkpoint)

        with pytest.raises(KeyError):
            loader.get_weight("nonexistent.weight")

    def test_iter_weights(self, sharded_checkpoint):
        """Test iterating over all weights."""
        loader = ShardedCheckpointLoader(sharded_checkpoint)

        weights = dict(loader.iter_weights())

        assert len(weights) == 4
        assert "blocks.0.q.weight" in weights
        assert "blocks.1.q.weight" in weights

    def test_clear_cache(self, sharded_checkpoint):
        """Test clearing loaded shard cache."""
        loader = ShardedCheckpointLoader(sharded_checkpoint)

        # Load a weight to populate cache
        loader.get_weight("blocks.0.q.weight")
        assert len(loader._loaded_shards) > 0

        # Clear cache
        loader.clear_cache()
        assert len(loader._loaded_shards) == 0


class TestConvertPytorchToMlx:
    """Tests for the main convert_pytorch_to_mlx function."""

    @pytest.fixture
    def simple_checkpoint(self, tmp_path):
        """Create a simple test checkpoint."""
        from safetensors.numpy import save_file

        weights = {
            "blocks.0.self_attn.q.weight": np.random.randn(128, 64).astype(np.float32),
            "blocks.0.self_attn.q.bias": np.random.randn(128).astype(np.float32),
            "blocks.0.self_attn.norm_q.weight": np.random.randn(128).astype(np.float32),
        }

        path = tmp_path / "model.safetensors"
        save_file(weights, str(path))
        return path

    def test_convert_single_file(self, simple_checkpoint, tmp_path):
        """Test converting a single safetensors file."""
        mlx_weights = convert_pytorch_to_mlx(
            simple_checkpoint,
            cache_dir=tmp_path,
            use_cache=False,
        )

        # Linear weight should be transposed
        assert mlx_weights["blocks.0.self_attn.q.weight"].shape == (64, 128)

        # Bias should keep original shape
        assert mlx_weights["blocks.0.self_attn.q.bias"].shape == (128,)

        # Norm weight should keep original shape
        assert mlx_weights["blocks.0.self_attn.norm_q.weight"].shape == (128,)

    def test_caching_works(self, simple_checkpoint, tmp_path):
        """Test that weights are cached and reloaded."""
        cache_dir = tmp_path / "cache"

        # First conversion
        mlx_weights1 = convert_pytorch_to_mlx(
            simple_checkpoint,
            cache_dir=cache_dir,
            use_cache=True,
        )

        # Check cache file exists
        cache_files = list(cache_dir.glob("*.safetensors"))
        assert len(cache_files) == 1

        # Second conversion should use cache
        mlx_weights2 = convert_pytorch_to_mlx(
            simple_checkpoint,
            cache_dir=cache_dir,
            use_cache=True,
        )

        # Values should be identical
        for name in mlx_weights1:
            mx.eval(mlx_weights1[name], mlx_weights2[name])
            np.testing.assert_array_equal(
                np.array(mlx_weights1[name]),
                np.array(mlx_weights2[name]),
            )


class TestLoadMlxWeights:
    """Tests for load_mlx_weights convenience function."""

    def test_load_mlx_weights(self, tmp_path):
        """Test the convenience function."""
        from safetensors.numpy import save_file

        weights = {
            "linear.weight": np.random.randn(32, 16).astype(np.float32),
        }

        path = tmp_path / "model.safetensors"
        save_file(weights, str(path))

        mlx_weights = load_mlx_weights(path, cache_dir=tmp_path)

        assert "linear.weight" in mlx_weights


class TestValidateWeightShapes:
    """Tests for validate_weight_shapes function."""

    def test_valid_shapes(self):
        """Test with matching shapes."""
        weights = {
            "weight1": mx.zeros((64, 32)),
            "weight2": mx.zeros((128,)),
        }
        expected = {
            "weight1": (64, 32),
            "weight2": (128,),
        }

        errors = validate_weight_shapes(weights, expected)
        assert len(errors) == 0

    def test_missing_weight(self):
        """Test with missing weight."""
        weights = {
            "weight1": mx.zeros((64, 32)),
        }
        expected = {
            "weight1": (64, 32),
            "weight2": (128,),
        }

        errors = validate_weight_shapes(weights, expected)
        assert len(errors) == 1
        assert "Missing" in errors[0]

    def test_shape_mismatch(self):
        """Test with shape mismatch."""
        weights = {
            "weight1": mx.zeros((32, 64)),  # Wrong shape
        }
        expected = {
            "weight1": (64, 32),
        }

        errors = validate_weight_shapes(weights, expected)
        assert len(errors) == 1
        assert "Shape mismatch" in errors[0]


class TestPyTorchModelConversion:
    """Integration tests converting actual PyTorch models."""

    def test_convert_linear_layer(self, tmp_path):
        """Test converting a PyTorch Linear layer with WanModel-style naming."""
        # Create PyTorch model
        torch_linear = nn.Linear(128, 256)

        # Save with WanModel-style naming to trigger transpose
        from safetensors.torch import save_file

        state_dict = {
            "blocks.0.self_attn.q.weight": torch_linear.weight,
            "blocks.0.self_attn.q.bias": torch_linear.bias,
        }
        path = tmp_path / "linear.safetensors"
        save_file(state_dict, str(path))

        # Convert to MLX
        mlx_weights = convert_pytorch_to_mlx(path, cache_dir=tmp_path, use_cache=False)

        # Verify shape transformation
        # PyTorch: [out_features, in_features] = [256, 128]
        # MLX: [in_features, out_features] = [128, 256]
        assert mlx_weights["blocks.0.self_attn.q.weight"].shape == (128, 256)
        assert mlx_weights["blocks.0.self_attn.q.bias"].shape == (256,)

        # Verify values by computing forward pass
        np.random.seed(42)
        test_input = np.random.randn(1, 128).astype(np.float32)

        # PyTorch forward
        torch_out = torch_linear(torch.from_numpy(test_input))
        torch_out_np = torch_out.detach().numpy()

        # MLX forward (manual matmul)
        mlx_input = mx.array(test_input)
        mx.eval(
            mlx_weights["blocks.0.self_attn.q.weight"],
            mlx_weights["blocks.0.self_attn.q.bias"],
        )
        mlx_out = (
            mlx_input @ mlx_weights["blocks.0.self_attn.q.weight"]
            + mlx_weights["blocks.0.self_attn.q.bias"]
        )
        mx.eval(mlx_out)
        mlx_out_np = np.array(mlx_out)

        # Should match
        np.testing.assert_allclose(mlx_out_np, torch_out_np, rtol=1e-5, atol=1e-5)

    def test_convert_attention_projection(self, tmp_path):
        """Test converting attention Q/K/V projections."""
        dim = 256
        num_heads = 4

        # Create PyTorch projections
        q_proj = nn.Linear(dim, dim)
        k_proj = nn.Linear(dim, dim)
        v_proj = nn.Linear(dim, dim)

        state_dict = {
            "blocks.0.self_attn.q.weight": q_proj.weight.data.numpy(),
            "blocks.0.self_attn.q.bias": q_proj.bias.data.numpy(),
            "blocks.0.self_attn.k.weight": k_proj.weight.data.numpy(),
            "blocks.0.self_attn.k.bias": k_proj.bias.data.numpy(),
            "blocks.0.self_attn.v.weight": v_proj.weight.data.numpy(),
            "blocks.0.self_attn.v.bias": v_proj.bias.data.numpy(),
        }

        converter = WeightConverter(cache_dir=tmp_path, use_cache=False)
        mlx_weights = converter.convert_state_dict(state_dict)

        # All projections should be transposed
        assert mlx_weights["blocks.0.self_attn.q.weight"].shape == (dim, dim)
        assert mlx_weights["blocks.0.self_attn.k.weight"].shape == (dim, dim)
        assert mlx_weights["blocks.0.self_attn.v.weight"].shape == (dim, dim)

        # Biases should not be transposed
        assert mlx_weights["blocks.0.self_attn.q.bias"].shape == (dim,)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_unsupported_format(self, tmp_path):
        """Test that unsupported formats raise error."""
        bad_file = tmp_path / "model.unsupported"
        bad_file.write_bytes(b"garbage")

        converter = WeightConverter(cache_dir=tmp_path)

        with pytest.raises(ValueError, match="Unsupported"):
            converter.load_checkpoint(bad_file)

    def test_missing_index_file(self, tmp_path):
        """Test that missing index file raises error."""
        # Create directory without index
        checkpoint_dir = tmp_path / "sharded"
        checkpoint_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="index.json"):
            ShardedCheckpointLoader(checkpoint_dir)

    def test_empty_state_dict(self, tmp_path):
        """Test converting empty state dict."""
        converter = WeightConverter(cache_dir=tmp_path)
        mlx_weights = converter.convert_state_dict({})

        assert len(mlx_weights) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
