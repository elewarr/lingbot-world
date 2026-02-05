"""Unit tests for MLX quantization support.

Tests for:
- Quantization types (int8, int4, nf4, mxfp4)
- Selective layer exclusion
- Memory savings verification
- Save/load quantized models
- Output quality after quantization
"""

import math
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pytest

import mlx.core as mx
import mlx.nn as nn_mlx
from mlx.utils import tree_flatten

from wan.backends.mlx.quantize import (
    quantize_model,
    save_quantized_model,
    load_quantized_model,
    get_model_memory_bytes,
    QuantizationConfig,
    DEFAULT_EXCLUDE_PATTERNS,
    QUANT_TYPE_MAP,
)


class SimpleLinearModel(nn_mlx.Module):
    """Simple test model with multiple linear layers."""

    def __init__(self, in_dim: int = 256, hidden_dim: int = 512, out_dim: int = 128):
        super().__init__()
        self.linear1 = nn_mlx.Linear(in_dim, hidden_dim)
        self.linear2 = nn_mlx.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn_mlx.Linear(hidden_dim, out_dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.linear1(x)
        x = nn_mlx.relu(x)
        x = self.linear2(x)
        x = nn_mlx.relu(x)
        return self.linear3(x)


class ModelWithEmbeddings(nn_mlx.Module):
    """Test model with embedding and head layers (mimics WanModel structure)."""

    def __init__(
        self,
        vocab_size: int = 1000,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_classes: int = 100,
    ):
        super().__init__()
        self.patch_embedding = nn_mlx.Linear(embed_dim, hidden_dim)  # Should be excluded
        self.block1 = nn_mlx.Linear(hidden_dim, hidden_dim)
        self.block2 = nn_mlx.Linear(hidden_dim, hidden_dim)
        self.head = SimpleHead(hidden_dim, num_classes)  # head.head should be excluded

    def __call__(self, x: mx.array) -> mx.array:
        x = self.patch_embedding(x)
        x = nn_mlx.relu(x)
        x = self.block1(x)
        x = nn_mlx.relu(x)
        x = self.block2(x)
        return self.head(x)


class SimpleHead(nn_mlx.Module):
    """Simple head module with a linear layer."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.head = nn_mlx.Linear(in_dim, out_dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.head(x)


class TestQuantizationConfig:
    """Tests for QuantizationConfig class."""

    def test_default_config(self):
        """Test default config uses int4."""
        config = QuantizationConfig()
        assert config.quant_type == 'int4'
        assert config.mode == 'affine'
        assert config.bits == 4
        assert config.group_size == 64

    def test_int8_config(self):
        """Test INT8 config."""
        config = QuantizationConfig(quant_type='int8')
        assert config.mode == 'affine'
        assert config.bits == 8
        assert config.group_size == 64

    def test_nf4_config(self):
        """Test NF4 config uses nvfp4 mode."""
        config = QuantizationConfig(quant_type='nf4')
        assert config.mode == 'nvfp4'
        # nvfp4 uses fixed bits/group_size
        assert config.bits is None
        assert config.group_size is None

    def test_invalid_quant_type(self):
        """Test that invalid quant_type raises error."""
        with pytest.raises(ValueError, match="Unknown quant_type"):
            QuantizationConfig(quant_type='invalid')

    def test_custom_exclude_layers(self):
        """Test custom exclude layers."""
        exclude = ['custom_layer', 'another_layer']
        config = QuantizationConfig(exclude_layers=exclude)
        assert config.exclude_layers == exclude

    def test_default_exclude_layers(self):
        """Test default exclude layers are used."""
        config = QuantizationConfig()
        assert 'patch_embedding' in config.exclude_layers
        assert 'head.head' in config.exclude_layers

    def test_to_dict_from_dict(self):
        """Test serialization round-trip."""
        config = QuantizationConfig(
            quant_type='int8',
            exclude_layers=['layer1', 'layer2'],
        )
        d = config.to_dict()
        config2 = QuantizationConfig.from_dict(d)

        assert config2.quant_type == config.quant_type
        assert config2.mode == config.mode
        assert config2.bits == config.bits
        assert config2.exclude_layers == config.exclude_layers


class TestGetModelMemory:
    """Tests for get_model_memory_bytes function."""

    def test_memory_calculation(self):
        """Test memory is calculated correctly."""
        model = SimpleLinearModel(in_dim=128, hidden_dim=256, out_dim=64)
        mx.eval(model.parameters())

        mem = get_model_memory_bytes(model)

        # Calculate expected memory
        # linear1: (128, 256) weight + (256,) bias = 128*256 + 256 = 33024 floats
        # linear2: (256, 256) weight + (256,) bias = 256*256 + 256 = 65792 floats
        # linear3: (256, 64) weight + (64,) bias = 256*64 + 64 = 16448 floats
        # Total: 33024 + 65792 + 16448 = 115264 floats = 461056 bytes
        expected = (128 * 256 + 256 + 256 * 256 + 256 + 256 * 64 + 64) * 4
        assert mem == expected

    def test_memory_is_positive(self):
        """Test that memory is always positive."""
        model = SimpleLinearModel()
        mx.eval(model.parameters())
        mem = get_model_memory_bytes(model)
        assert mem > 0


class TestQuantizeModel:
    """Tests for quantize_model function."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        model = SimpleLinearModel(in_dim=128, hidden_dim=256, out_dim=64)
        mx.eval(model.parameters())
        return model

    def test_int8_quantization(self, simple_model):
        """Test INT8 quantization."""
        mem_before = get_model_memory_bytes(simple_model)

        quantize_model(simple_model, quant_type='int8')

        mem_after = get_model_memory_bytes(simple_model)
        reduction = (1 - mem_after / mem_before) * 100

        # INT8 should give ~50-75% reduction
        assert reduction > 40, f"Expected >40% reduction, got {reduction:.1f}%"
        assert isinstance(simple_model.linear1, nn_mlx.QuantizedLinear)

    def test_int4_quantization(self, simple_model):
        """Test INT4 quantization."""
        mem_before = get_model_memory_bytes(simple_model)

        quantize_model(simple_model, quant_type='int4')

        mem_after = get_model_memory_bytes(simple_model)
        reduction = (1 - mem_after / mem_before) * 100

        # INT4 should give ~75-85% reduction
        assert reduction > 70, f"Expected >70% reduction, got {reduction:.1f}%"
        assert isinstance(simple_model.linear1, nn_mlx.QuantizedLinear)

    def test_nf4_quantization(self, simple_model):
        """Test NF4 (nvfp4) quantization."""
        mem_before = get_model_memory_bytes(simple_model)

        quantize_model(simple_model, quant_type='nf4')

        mem_after = get_model_memory_bytes(simple_model)
        reduction = (1 - mem_after / mem_before) * 100

        # NF4 should give ~75-85% reduction
        assert reduction > 70, f"Expected >70% reduction, got {reduction:.1f}%"

    def test_mxfp4_quantization(self, simple_model):
        """Test MXFP4 quantization."""
        mem_before = get_model_memory_bytes(simple_model)

        quantize_model(simple_model, quant_type='mxfp4')

        mem_after = get_model_memory_bytes(simple_model)
        reduction = (1 - mem_after / mem_before) * 100

        # MXFP4 should give ~75-85% reduction
        assert reduction > 70, f"Expected >70% reduction, got {reduction:.1f}%"

    def test_stores_config_on_model(self, simple_model):
        """Test that config is stored on model."""
        quantize_model(simple_model, quant_type='int4')

        assert hasattr(simple_model, '_quantization_config')
        config = simple_model._quantization_config
        assert config.quant_type == 'int4'

    def test_model_still_works_after_quantization(self, simple_model):
        """Test that model produces valid output after quantization."""
        x = mx.random.normal((4, 128))

        # Get output before quantization
        y_before = simple_model(x)
        mx.eval(y_before)

        # Quantize and get output
        quantize_model(simple_model, quant_type='int4')
        y_after = simple_model(x)
        mx.eval(y_after)

        # Output shape should be the same
        assert y_after.shape == y_before.shape
        # Output should be finite
        assert not mx.isnan(y_after).any()
        assert not mx.isinf(y_after).any()


class TestSelectiveQuantization:
    """Tests for selective layer exclusion."""

    @pytest.fixture
    def model_with_embeddings(self):
        """Create a model with embeddings for testing."""
        model = ModelWithEmbeddings()
        mx.eval(model.parameters())
        return model

    def test_default_exclusions(self, model_with_embeddings):
        """Test that default exclusions work (patch_embedding, head.head)."""
        quantize_model(model_with_embeddings, quant_type='int4')

        # patch_embedding should NOT be quantized
        assert isinstance(model_with_embeddings.patch_embedding, nn_mlx.Linear)
        assert not isinstance(model_with_embeddings.patch_embedding, nn_mlx.QuantizedLinear)

        # head.head should NOT be quantized
        assert isinstance(model_with_embeddings.head.head, nn_mlx.Linear)
        assert not isinstance(model_with_embeddings.head.head, nn_mlx.QuantizedLinear)

        # block1 and block2 SHOULD be quantized
        assert isinstance(model_with_embeddings.block1, nn_mlx.QuantizedLinear)
        assert isinstance(model_with_embeddings.block2, nn_mlx.QuantizedLinear)

    def test_custom_exclusions(self, model_with_embeddings):
        """Test custom exclusion patterns."""
        # Exclude only block1, allow everything else
        quantize_model(
            model_with_embeddings,
            quant_type='int4',
            exclude_layers=['block1'],
        )

        # block1 should NOT be quantized
        assert isinstance(model_with_embeddings.block1, nn_mlx.Linear)
        assert not isinstance(model_with_embeddings.block1, nn_mlx.QuantizedLinear)

        # Everything else SHOULD be quantized (default exclusions not applied)
        assert isinstance(model_with_embeddings.patch_embedding, nn_mlx.QuantizedLinear)
        assert isinstance(model_with_embeddings.block2, nn_mlx.QuantizedLinear)
        assert isinstance(model_with_embeddings.head.head, nn_mlx.QuantizedLinear)

    def test_no_exclusions(self, model_with_embeddings):
        """Test with no exclusions (empty list)."""
        quantize_model(
            model_with_embeddings,
            quant_type='int4',
            exclude_layers=[],
        )

        # Everything should be quantized
        assert isinstance(model_with_embeddings.patch_embedding, nn_mlx.QuantizedLinear)
        assert isinstance(model_with_embeddings.block1, nn_mlx.QuantizedLinear)
        assert isinstance(model_with_embeddings.block2, nn_mlx.QuantizedLinear)
        assert isinstance(model_with_embeddings.head.head, nn_mlx.QuantizedLinear)


class TestSaveLoadQuantizedModel:
    """Tests for save/load quantized model functionality."""

    @pytest.fixture
    def quantized_model(self):
        """Create a quantized model for testing."""
        model = SimpleLinearModel(in_dim=128, hidden_dim=256, out_dim=64)
        mx.eval(model.parameters())
        quantize_model(model, quant_type='int4')
        return model

    def test_save_and_load_directory(self, quantized_model):
        """Test saving and loading from directory."""
        x = mx.random.normal((4, 128))
        y_expected = quantized_model(x)
        mx.eval(y_expected)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            save_quantized_model(quantized_model, tmpdir)

            # Check files exist
            assert Path(tmpdir, 'weights.safetensors').exists()
            assert Path(tmpdir, 'quantization_config.json').exists()

            # Load into new model
            new_model = SimpleLinearModel(in_dim=128, hidden_dim=256, out_dim=64)
            mx.eval(new_model.parameters())
            loaded_model = load_quantized_model(tmpdir, model=new_model)

            # Verify output matches
            y_loaded = loaded_model(x)
            mx.eval(y_loaded)

            np.testing.assert_allclose(
                np.array(y_expected),
                np.array(y_loaded),
                rtol=1e-5,
                atol=1e-5,
            )

    def test_save_and_load_safetensors_file(self, quantized_model):
        """Test saving and loading from .safetensors file."""
        x = mx.random.normal((4, 128))
        y_expected = quantized_model(x)
        mx.eval(y_expected)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'model.safetensors'

            # Save
            save_quantized_model(quantized_model, path)

            # Check files exist
            assert path.exists()
            assert path.with_suffix('.json').exists()

            # Load into new model
            new_model = SimpleLinearModel(in_dim=128, hidden_dim=256, out_dim=64)
            mx.eval(new_model.parameters())
            loaded_model = load_quantized_model(path, model=new_model)

            # Verify output matches
            y_loaded = loaded_model(x)
            mx.eval(y_loaded)

            np.testing.assert_allclose(
                np.array(y_expected),
                np.array(y_loaded),
                rtol=1e-5,
                atol=1e-5,
            )

    def test_load_with_model_cls(self, quantized_model):
        """Test loading using model_cls parameter."""
        x = mx.random.normal((4, 128))
        y_expected = quantized_model(x)
        mx.eval(y_expected)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_quantized_model(quantized_model, tmpdir)

            # Load using model_cls
            loaded_model = load_quantized_model(
                tmpdir,
                model_cls=SimpleLinearModel,
                model_kwargs={'in_dim': 128, 'hidden_dim': 256, 'out_dim': 64},
            )

            # Verify quantization was applied
            assert isinstance(loaded_model.linear1, nn_mlx.QuantizedLinear)

            # Verify output matches
            y_loaded = loaded_model(x)
            mx.eval(y_loaded)

            np.testing.assert_allclose(
                np.array(y_expected),
                np.array(y_loaded),
                rtol=1e-5,
                atol=1e-5,
            )

    def test_load_requires_model_or_cls(self):
        """Test that load requires either model or model_cls."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a dummy file
            Path(tmpdir, 'weights.safetensors').touch()

            with pytest.raises(ValueError, match="Either model or model_cls"):
                load_quantized_model(tmpdir)

    def test_config_is_preserved(self, quantized_model):
        """Test that quantization config is preserved after load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_quantized_model(quantized_model, tmpdir)

            new_model = SimpleLinearModel(in_dim=128, hidden_dim=256, out_dim=64)
            mx.eval(new_model.parameters())
            loaded_model = load_quantized_model(tmpdir, model=new_model)

            assert hasattr(loaded_model, '_quantization_config')
            config = loaded_model._quantization_config
            assert config.quant_type == 'int4'


class TestMemorySavings:
    """Tests that verify expected memory savings for each quantization type."""

    @pytest.fixture
    def large_model(self):
        """Create a larger model for more accurate memory measurements."""
        # Larger model to reduce overhead effects
        model = SimpleLinearModel(in_dim=1024, hidden_dim=2048, out_dim=512)
        mx.eval(model.parameters())
        return model

    def test_int8_memory_savings(self, large_model):
        """INT8 should achieve ~50% or better memory reduction."""
        mem_before = get_model_memory_bytes(large_model)

        quantize_model(large_model, quant_type='int8')

        mem_after = get_model_memory_bytes(large_model)
        reduction = (1 - mem_after / mem_before) * 100

        # INT8 typically gives 50-75% reduction
        # (8 bits + scale overhead vs 32 bits)
        assert reduction >= 45, f"INT8 should give >=45% reduction, got {reduction:.1f}%"

    def test_int4_memory_savings(self, large_model):
        """INT4 should achieve ~75% or better memory reduction."""
        mem_before = get_model_memory_bytes(large_model)

        quantize_model(large_model, quant_type='int4')

        mem_after = get_model_memory_bytes(large_model)
        reduction = (1 - mem_after / mem_before) * 100

        # INT4 typically gives 75-85% reduction
        assert reduction >= 70, f"INT4 should give >=70% reduction, got {reduction:.1f}%"

    def test_nf4_memory_savings(self, large_model):
        """NF4 should achieve ~75% or better memory reduction."""
        mem_before = get_model_memory_bytes(large_model)

        quantize_model(large_model, quant_type='nf4')

        mem_after = get_model_memory_bytes(large_model)
        reduction = (1 - mem_after / mem_before) * 100

        # NF4 (nvfp4) typically gives 75-85% reduction
        assert reduction >= 70, f"NF4 should give >=70% reduction, got {reduction:.1f}%"


class TestOutputQuality:
    """Tests that verify output quality after quantization."""

    @pytest.fixture
    def model_and_input(self):
        """Create model and test input."""
        model = SimpleLinearModel(in_dim=256, hidden_dim=512, out_dim=128)
        mx.eval(model.parameters())
        # Use deterministic initialization with correct shapes
        # MLX Linear weight shape: [out_features, in_features]
        mx.random.seed(42)
        model.linear1.weight = mx.random.normal((512, 256)) * 0.02
        model.linear2.weight = mx.random.normal((512, 512)) * 0.02
        model.linear3.weight = mx.random.normal((128, 512)) * 0.02
        mx.eval(model.parameters())

        x = mx.random.normal((8, 256))
        return model, x

    def test_int8_output_quality(self, model_and_input):
        """INT8 quantization should preserve output within tolerance."""
        model, x = model_and_input

        # Get reference output
        y_ref = model(x)
        mx.eval(y_ref)
        y_ref_np = np.array(y_ref)

        # Create fresh model and quantize
        model2 = SimpleLinearModel(in_dim=256, hidden_dim=512, out_dim=128)
        # Use tree_flatten to get flat list of (key, value) pairs
        flat_weights = tree_flatten(model.parameters())
        model2.load_weights(flat_weights)
        mx.eval(model2.parameters())

        quantize_model(model2, quant_type='int8')
        y_quant = model2(x)
        mx.eval(y_quant)
        y_quant_np = np.array(y_quant)

        # INT8 should be quite close to original
        # Allow some tolerance due to quantization error
        rel_error = np.abs(y_quant_np - y_ref_np) / (np.abs(y_ref_np) + 1e-8)
        mean_rel_error = np.mean(rel_error)

        assert mean_rel_error < 0.5, f"INT8 mean relative error too high: {mean_rel_error:.4f}"

    def test_int4_output_reasonable(self, model_and_input):
        """INT4 quantization should produce reasonable (finite, not NaN) output."""
        model, x = model_and_input

        quantize_model(model, quant_type='int4')
        y = model(x)
        mx.eval(y)

        # Output should be finite
        assert not mx.isnan(y).any(), "INT4 output contains NaN"
        assert not mx.isinf(y).any(), "INT4 output contains Inf"
        assert y.shape == (8, 128), f"Unexpected output shape: {y.shape}"

    def test_nf4_output_reasonable(self, model_and_input):
        """NF4 quantization should produce reasonable output."""
        model, x = model_and_input

        quantize_model(model, quant_type='nf4')
        y = model(x)
        mx.eval(y)

        # Output should be finite
        assert not mx.isnan(y).any(), "NF4 output contains NaN"
        assert not mx.isinf(y).any(), "NF4 output contains Inf"


class TestDefaultExcludePatterns:
    """Tests for default exclude patterns."""

    def test_default_patterns_defined(self):
        """Test that default patterns are defined."""
        assert len(DEFAULT_EXCLUDE_PATTERNS) > 0
        assert 'patch_embedding' in DEFAULT_EXCLUDE_PATTERNS
        assert 'head.head' in DEFAULT_EXCLUDE_PATTERNS

    def test_patterns_used_by_default(self):
        """Test that default patterns are used by QuantizationConfig."""
        config = QuantizationConfig()
        for pattern in DEFAULT_EXCLUDE_PATTERNS:
            assert pattern in config.exclude_layers


class TestQuantTypeMap:
    """Tests for QUANT_TYPE_MAP."""

    def test_all_types_defined(self):
        """Test that all expected types are in the map."""
        expected_types = ['int8', 'int4', 'nf4', 'mxfp4']
        for qtype in expected_types:
            assert qtype in QUANT_TYPE_MAP

    def test_all_types_have_mode(self):
        """Test that all types have mode defined."""
        for qtype, params in QUANT_TYPE_MAP.items():
            assert 'mode' in params
            assert params['mode'] in ['affine', 'nvfp4', 'mxfp4']
