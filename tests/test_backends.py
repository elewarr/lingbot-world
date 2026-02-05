"""Unit tests for backend abstraction layer.

Tests backend registry, tensor transfer utilities, and backend implementations.
"""

import numpy as np
import pytest
import sys

# Import frameworks
import torch

# Import backends
from wan.backends.base import (
    Backend,
    register_backend,
    get_backend,
    list_backends,
    get_available_backends,
    get_default_backend,
    mx_to_torch,
    torch_to_mx,
    mx_to_torch_list,
    torch_to_mx_list,
    convert_dit_cond_dict_to_mx,
    convert_dit_cond_dict_to_torch,
)
from wan.backends.pytorch import PyTorchBackend


# Check if MLX is available
IS_MACOS = sys.platform == 'darwin'
HAS_MLX = False
if IS_MACOS:
    try:
        import mlx.core as mx
        HAS_MLX = True
    except ImportError:
        pass


class TestBackendRegistry:
    """Tests for backend registry functions."""

    def test_list_backends(self):
        """Test that backend list includes expected backends."""
        backends = list_backends()
        assert 'pytorch' in backends
        # MLX should be registered on macOS
        if IS_MACOS and HAS_MLX:
            assert 'mlx' in backends

    def test_get_available_backends(self):
        """Test that available backends can be instantiated."""
        available = get_available_backends()
        assert 'pytorch' in available  # PyTorch should always be available
        
        # MLX only available on macOS with Apple Silicon
        if IS_MACOS and HAS_MLX:
            assert 'mlx' in available

    def test_get_default_backend(self):
        """Test default backend selection."""
        default = get_default_backend()
        # PyTorch should be preferred for backward compatibility
        assert default == 'pytorch'

    def test_get_backend_pytorch(self):
        """Test getting PyTorch backend."""
        backend = get_backend('pytorch')
        assert isinstance(backend, PyTorchBackend)
        assert backend.name == 'pytorch'
        assert backend.is_available

    def test_get_backend_unknown(self):
        """Test error on unknown backend."""
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend('unknown_backend')

    @pytest.mark.skipif(not (IS_MACOS and HAS_MLX), reason="MLX not available")
    def test_get_backend_mlx(self):
        """Test getting MLX backend."""
        from wan.backends.mlx.backend import MLXBackend
        backend = get_backend('mlx')
        assert isinstance(backend, MLXBackend)
        assert backend.name == 'mlx'
        assert backend.is_available


class TestTensorTransfer:
    """Tests for tensor transfer utilities."""

    @pytest.mark.skipif(not (IS_MACOS and HAS_MLX), reason="MLX not available")
    def test_torch_to_mx_simple(self):
        """Test converting PyTorch tensor to MLX array."""
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        arr = torch_to_mx(tensor)
        
        mx.eval(arr)
        np.testing.assert_allclose(np.array(arr), tensor.numpy())

    @pytest.mark.skipif(not (IS_MACOS and HAS_MLX), reason="MLX not available")
    def test_mx_to_torch_simple(self):
        """Test converting MLX array to PyTorch tensor."""
        arr = mx.array([1.0, 2.0, 3.0, 4.0])
        tensor = mx_to_torch(arr)
        
        np.testing.assert_allclose(tensor.numpy(), np.array(arr))

    @pytest.mark.skipif(not (IS_MACOS and HAS_MLX), reason="MLX not available")
    def test_torch_to_mx_with_grad(self):
        """Test converting tensor with gradient history."""
        tensor = torch.tensor([1.0, 2.0], requires_grad=True)
        tensor = tensor * 2  # Create computation graph
        
        # Should handle detaching automatically
        arr = torch_to_mx(tensor)
        mx.eval(arr)
        np.testing.assert_allclose(np.array(arr), [2.0, 4.0])

    @pytest.mark.skipif(not (IS_MACOS and HAS_MLX), reason="MLX not available")
    def test_roundtrip_torch_mx_torch(self):
        """Test torch -> mlx -> torch roundtrip."""
        original = torch.randn(4, 16, 32, dtype=torch.float32)
        
        arr = torch_to_mx(original)
        recovered = mx_to_torch(arr)
        
        np.testing.assert_allclose(
            recovered.numpy(), original.numpy(), rtol=1e-6
        )

    @pytest.mark.skipif(not (IS_MACOS and HAS_MLX), reason="MLX not available")
    def test_mx_to_torch_with_device(self):
        """Test converting with target device."""
        arr = mx.array([1.0, 2.0, 3.0])
        tensor = mx_to_torch(arr, device=torch.device('cpu'))
        
        assert tensor.device == torch.device('cpu')
        np.testing.assert_allclose(tensor.numpy(), np.array(arr))

    @pytest.mark.skipif(not (IS_MACOS and HAS_MLX), reason="MLX not available")
    def test_list_conversion(self):
        """Test converting lists of tensors/arrays."""
        tensors = [torch.randn(4, 8) for _ in range(3)]
        
        arrs = torch_to_mx_list(tensors)
        assert len(arrs) == 3
        
        recovered = mx_to_torch_list(arrs)
        assert len(recovered) == 3
        
        for orig, rec in zip(tensors, recovered):
            np.testing.assert_allclose(rec.numpy(), orig.numpy(), rtol=1e-6)

    @pytest.mark.skipif(not (IS_MACOS and HAS_MLX), reason="MLX not available")
    def test_dit_cond_dict_conversion(self):
        """Test converting camera conditioning dict."""
        # Create PyTorch dict
        torch_dict = {
            "c2ws_plucker_emb": [torch.randn(1, 100, 384) for _ in range(2)],
            "other_key": "some_value",  # Non-tensor should pass through
        }
        
        # Convert to MLX
        mlx_dict = convert_dit_cond_dict_to_mx(torch_dict)
        assert len(mlx_dict["c2ws_plucker_emb"]) == 2
        assert mlx_dict["other_key"] == "some_value"
        
        # Convert back to torch
        recovered = convert_dit_cond_dict_to_torch(mlx_dict)
        assert len(recovered["c2ws_plucker_emb"]) == 2
        
        for orig, rec in zip(torch_dict["c2ws_plucker_emb"], recovered["c2ws_plucker_emb"]):
            np.testing.assert_allclose(rec.numpy(), orig.numpy(), rtol=1e-6)

    @pytest.mark.skipif(not (IS_MACOS and HAS_MLX), reason="MLX not available")
    def test_dit_cond_dict_none(self):
        """Test handling None dict."""
        assert convert_dit_cond_dict_to_mx(None) is None
        assert convert_dit_cond_dict_to_torch(None) is None


class TestPyTorchBackend:
    """Tests for PyTorchBackend."""

    def test_name(self):
        """Test backend name."""
        backend = PyTorchBackend()
        assert backend.name == 'pytorch'

    def test_is_available(self):
        """Test availability check."""
        backend = PyTorchBackend()
        assert backend.is_available

    def test_to_numpy(self):
        """Test tensor to numpy conversion."""
        backend = PyTorchBackend()
        tensor = torch.randn(4, 8, 16)
        
        arr = backend.to_numpy(tensor)
        
        assert isinstance(arr, np.ndarray)
        np.testing.assert_allclose(arr, tensor.numpy())

    def test_from_numpy(self):
        """Test numpy to tensor conversion."""
        backend = PyTorchBackend()
        arr = np.random.randn(4, 8, 16).astype(np.float32)
        
        tensor = backend.from_numpy(arr)
        
        assert isinstance(tensor, torch.Tensor)
        np.testing.assert_allclose(tensor.numpy(), arr)

    def test_from_numpy_with_dtype(self):
        """Test numpy conversion with dtype override."""
        backend = PyTorchBackend()
        arr = np.random.randn(4, 8).astype(np.float32)
        
        tensor = backend.from_numpy(arr, dtype=torch.float16)
        
        assert tensor.dtype == torch.float16

    def test_from_numpy_with_device(self):
        """Test numpy conversion with device specification."""
        backend = PyTorchBackend()
        arr = np.random.randn(4, 8).astype(np.float32)
        
        tensor = backend.from_numpy(arr, device='cpu')
        
        assert tensor.device == torch.device('cpu')


@pytest.mark.skipif(not (IS_MACOS and HAS_MLX), reason="MLX not available")
class TestMLXBackend:
    """Tests for MLXBackend."""

    def test_name(self):
        """Test backend name."""
        from wan.backends.mlx.backend import MLXBackend
        backend = MLXBackend()
        assert backend.name == 'mlx'

    def test_is_available(self):
        """Test availability check."""
        from wan.backends.mlx.backend import MLXBackend
        backend = MLXBackend()
        assert backend.is_available

    def test_to_numpy(self):
        """Test array to numpy conversion."""
        from wan.backends.mlx.backend import MLXBackend
        backend = MLXBackend()
        arr = mx.random.normal((4, 8, 16))
        mx.eval(arr)
        
        np_arr = backend.to_numpy(arr)
        
        assert isinstance(np_arr, np.ndarray)
        np.testing.assert_allclose(np_arr, np.array(arr))

    def test_from_numpy(self):
        """Test numpy to array conversion."""
        from wan.backends.mlx.backend import MLXBackend
        backend = MLXBackend()
        np_arr = np.random.randn(4, 8, 16).astype(np.float32)
        
        arr = backend.from_numpy(np_arr)
        
        mx.eval(arr)
        np.testing.assert_allclose(np.array(arr), np_arr)

    def test_from_numpy_with_dtype(self):
        """Test numpy conversion with dtype override."""
        from wan.backends.mlx.backend import MLXBackend
        backend = MLXBackend()
        np_arr = np.random.randn(4, 8).astype(np.float32)
        
        arr = backend.from_numpy(np_arr, dtype=mx.float16)
        mx.eval(arr)
        
        assert arr.dtype == mx.float16


class TestBackendAbstraction:
    """Tests for Backend ABC contract."""

    def test_pytorch_backend_implements_interface(self):
        """Test that PyTorchBackend implements all abstract methods."""
        backend = PyTorchBackend()
        
        # Check all abstract methods are implemented
        assert hasattr(backend, 'name')
        assert hasattr(backend, 'is_available')
        assert hasattr(backend, 'load_model')
        assert hasattr(backend, 'forward')
        assert hasattr(backend, 'to_numpy')
        assert hasattr(backend, 'from_numpy')

    @pytest.mark.skipif(not (IS_MACOS and HAS_MLX), reason="MLX not available")
    def test_mlx_backend_implements_interface(self):
        """Test that MLXBackend implements all abstract methods."""
        from wan.backends.mlx.backend import MLXBackend
        backend = MLXBackend()
        
        # Check all abstract methods are implemented
        assert hasattr(backend, 'name')
        assert hasattr(backend, 'is_available')
        assert hasattr(backend, 'load_model')
        assert hasattr(backend, 'forward')
        assert hasattr(backend, 'to_numpy')
        assert hasattr(backend, 'from_numpy')


class TestTensorTransferDtypes:
    """Tests for dtype preservation in tensor transfer."""

    @pytest.mark.skipif(not (IS_MACOS and HAS_MLX), reason="MLX not available")
    def test_float32_preservation(self):
        """Test float32 dtype is preserved."""
        tensor = torch.randn(4, 8, dtype=torch.float32)
        arr = torch_to_mx(tensor)
        mx.eval(arr)
        
        # MLX should maintain float32
        assert arr.dtype == mx.float32
        
        recovered = mx_to_torch(arr)
        assert recovered.dtype == torch.float32

    @pytest.mark.skipif(not (IS_MACOS and HAS_MLX), reason="MLX not available")
    def test_float16_conversion(self):
        """Test float16 conversion."""
        tensor = torch.randn(4, 8, dtype=torch.float16)
        arr = torch_to_mx(tensor)
        mx.eval(arr)
        
        # MLX should preserve float16
        assert arr.dtype == mx.float16

    @pytest.mark.skipif(not (IS_MACOS and HAS_MLX), reason="MLX not available")
    def test_int_types(self):
        """Test integer type handling."""
        tensor = torch.randint(0, 100, (4, 8), dtype=torch.int32)
        arr = torch_to_mx(tensor)
        mx.eval(arr)
        
        np.testing.assert_array_equal(np.array(arr), tensor.numpy())


class TestTensorTransferShapes:
    """Tests for shape preservation in tensor transfer."""

    @pytest.mark.skipif(not (IS_MACOS and HAS_MLX), reason="MLX not available")
    def test_1d_tensor(self):
        """Test 1D tensor transfer."""
        tensor = torch.randn(100)
        arr = torch_to_mx(tensor)
        mx.eval(arr)
        
        assert arr.shape == (100,)

    @pytest.mark.skipif(not (IS_MACOS and HAS_MLX), reason="MLX not available")
    def test_4d_tensor(self):
        """Test 4D tensor transfer (video latent shape)."""
        tensor = torch.randn(16, 21, 60, 104)  # [C, F, H, W]
        arr = torch_to_mx(tensor)
        mx.eval(arr)
        
        assert arr.shape == (16, 21, 60, 104)

    @pytest.mark.skipif(not (IS_MACOS and HAS_MLX), reason="MLX not available")
    def test_5d_tensor(self):
        """Test 5D tensor transfer (batch video)."""
        tensor = torch.randn(1, 16, 21, 60, 104)  # [B, C, F, H, W]
        arr = torch_to_mx(tensor)
        mx.eval(arr)
        
        assert arr.shape == (1, 16, 21, 60, 104)
