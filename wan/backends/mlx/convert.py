"""Weight converter for PyTorch to MLX format.

This module converts PyTorch WanModel checkpoints to MLX-compatible format,
handling the necessary weight transpositions and caching for efficient reuse.

Key transformations:
1. Linear weights: transpose [out_features, in_features] -> [in_features, out_features]
2. Convert torch.Tensor / numpy -> mx.array
3. Handle dtype conversion (bfloat16/float16 -> float32 for MPS compatibility)
4. Cache to ~/.cache/lingbot-world/mlx/{model_name}.safetensors

Supports:
- Single-file .safetensors checkpoints
- Sharded multi-file safetensors with index JSON
- Single-file .pth checkpoints
- Lazy loading for memory efficiency with large models
"""

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union

import mlx.core as mx
import numpy as np

__all__ = [
    "convert_pytorch_to_mlx",
    "load_mlx_weights",
    "get_cache_path",
    "is_linear_weight",
    "needs_transpose",
    "convert_weight",
    "WeightConverter",
    "ShardedCheckpointLoader",
]

logger = logging.getLogger(__name__)

# Default cache directory
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "lingbot-world" / "mlx"

# Patterns for identifying linear weights that need transpose
# Linear weight shape: [out_features, in_features] in PyTorch
# Linear weight shape: [in_features, out_features] in MLX
LINEAR_WEIGHT_PATTERNS = [
    # Attention projections
    r"\.q\.weight$",
    r"\.k\.weight$",
    r"\.v\.weight$",
    r"\.o\.weight$",
    # FFN layers
    r"\.ffn\.\d+\.weight$",
    r"\.ffn_linear\d*\.weight$",
    # Text/Time embeddings
    r"text_embedding\.\d+\.weight$",
    r"time_embedding\.\d+\.weight$",
    r"time_projection\.\d+\.weight$",
    # Camera control layers
    r"cam_injector_layer\d*\.weight$",
    r"cam_scale_layer\.weight$",
    r"cam_shift_layer\.weight$",
    r"c2ws_hidden_states_layer\d*\.weight$",
    r"patch_embedding_wancamctrl\.weight$",
    # Head
    r"head\.head\.weight$",
]

# Patterns for weights that should NOT be transposed
NO_TRANSPOSE_PATTERNS = [
    r"\.bias$",  # All biases
    r"\.modulation$",  # Modulation parameters
    r"\.norm\d*\.weight$",  # Norm layers (RMSNorm, LayerNorm)
    r"norm_q\.weight$",  # QK normalization
    r"norm_k\.weight$",
    r"norm3\.weight$",  # Cross-attention norm
    r"patch_embedding\.weight$",  # Conv3d weight (not Linear)
]

# Compiled regex patterns for efficiency
_LINEAR_PATTERNS = [re.compile(p) for p in LINEAR_WEIGHT_PATTERNS]
_NO_TRANSPOSE_PATTERNS = [re.compile(p) for p in NO_TRANSPOSE_PATTERNS]


def is_linear_weight(name: str) -> bool:
    """Check if a parameter name corresponds to a Linear layer weight.

    Args:
        name: Parameter name (e.g., 'blocks.0.self_attn.q.weight')

    Returns:
        True if this is a Linear weight that needs transpose
    """
    # First check if it's explicitly excluded
    for pattern in _NO_TRANSPOSE_PATTERNS:
        if pattern.search(name):
            return False

    # Then check if it matches a linear weight pattern
    for pattern in _LINEAR_PATTERNS:
        if pattern.search(name):
            return True

    return False


def needs_transpose(name: str, shape: Tuple[int, ...]) -> bool:
    """Determine if a weight tensor needs to be transposed.

    Linear layers in PyTorch have weights of shape [out_features, in_features],
    while MLX expects [in_features, out_features]. This function determines
    if a weight should be transposed based on its name and shape.

    Args:
        name: Parameter name
        shape: Tensor shape

    Returns:
        True if the weight should be transposed
    """
    # Only 2D weights can be transposed
    if len(shape) != 2:
        return False

    return is_linear_weight(name)


def convert_weight(
    weight: np.ndarray,
    name: str,
    target_dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """Convert a single weight tensor from PyTorch/numpy format to MLX.

    Args:
        weight: Numpy array (from PyTorch tensor)
        name: Parameter name for determining if transpose is needed
        target_dtype: Target MLX dtype (default: float32 for MPS compatibility)

    Returns:
        MLX array with appropriate shape and dtype
    """
    # Ensure numpy array
    if not isinstance(weight, np.ndarray):
        weight = np.array(weight)

    # Convert bfloat16 to float32 (numpy doesn't support bfloat16 natively)
    if weight.dtype == np.dtype("float16"):
        weight = weight.astype(np.float32)

    # Check if transpose is needed
    if needs_transpose(name, weight.shape):
        weight = weight.T

    # Convert to MLX array
    mlx_weight = mx.array(weight)

    # Cast to target dtype if needed
    if mlx_weight.dtype != target_dtype:
        mlx_weight = mlx_weight.astype(target_dtype)

    return mlx_weight


def compute_checkpoint_hash(checkpoint_path: Union[str, Path]) -> str:
    """Compute a hash for the checkpoint to use as cache key.

    For sharded checkpoints, hashes the index file.
    For single files, hashes file metadata (name + size + mtime).

    Args:
        checkpoint_path: Path to checkpoint directory or file

    Returns:
        SHA256 hash string (first 16 chars)
    """
    checkpoint_path = Path(checkpoint_path)
    hasher = hashlib.sha256()

    if checkpoint_path.is_dir():
        # Look for index file in sharded checkpoint
        index_files = list(checkpoint_path.glob("*.index.json"))
        if index_files:
            # Hash the index file content
            index_file = sorted(index_files)[0]
            with open(index_file, "rb") as fh:
                hasher.update(fh.read())
        else:
            # Hash directory listing
            for file_path in sorted(checkpoint_path.glob("*.safetensors")):
                hasher.update(file_path.name.encode())
                hasher.update(str(file_path.stat().st_size).encode())
    else:
        # Single file - hash name + size + mtime
        stat = checkpoint_path.stat()
        hasher.update(checkpoint_path.name.encode())
        hasher.update(str(stat.st_size).encode())
        hasher.update(str(int(stat.st_mtime)).encode())

    return hasher.hexdigest()[:16]


def get_cache_path(
    checkpoint_path: Union[str, Path],
    model_name: Optional[str] = None,
    cache_dir: Optional[Path] = None,
) -> Path:
    """Get the cache path for converted MLX weights.

    Args:
        checkpoint_path: Path to original checkpoint
        model_name: Optional model name for cache file
        cache_dir: Optional custom cache directory

    Returns:
        Path to cached MLX weights
    """
    checkpoint_path = Path(checkpoint_path)
    cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR

    if model_name is None:
        if checkpoint_path.is_dir():
            model_name = checkpoint_path.name
        else:
            model_name = checkpoint_path.stem

    # Include hash for cache invalidation
    ckpt_hash = compute_checkpoint_hash(checkpoint_path)
    cache_name = f"{model_name}_{ckpt_hash}.safetensors"

    return cache_dir / cache_name


class ShardedCheckpointLoader:
    """Lazy loader for sharded safetensors checkpoints.

    Loads weights on-demand from sharded checkpoint files,
    minimizing memory usage for large models.
    """

    def __init__(self, checkpoint_dir: Union[str, Path]):
        """Initialize the loader.

        Args:
            checkpoint_dir: Directory containing sharded checkpoint files
        """
        self.checkpoint_dir = Path(checkpoint_dir)

        # Load the index file
        index_files = list(self.checkpoint_dir.glob("*.index.json"))
        if not index_files:
            raise FileNotFoundError(f"No index.json found in {checkpoint_dir}")

        self.index_path = sorted(index_files)[0]
        with open(self.index_path, "r") as f:
            self.index = json.load(f)

        self.weight_map: Dict[str, str] = self.index.get("weight_map", {})
        self.metadata = self.index.get("metadata", {})

        # Cache of loaded shard files
        self._loaded_shards: Dict[str, Dict[str, np.ndarray]] = {}

    @property
    def total_size(self) -> int:
        """Total size of all weights in bytes."""
        size: int = self.metadata.get("total_size", 0)
        return size

    @property
    def parameter_names(self) -> List[str]:
        """List of all parameter names."""
        return list(self.weight_map.keys())

    @property
    def shard_files(self) -> Set[str]:
        """Set of unique shard filenames."""
        return set(self.weight_map.values())

    def get_shard_path(self, shard_name: str) -> Path:
        """Get full path to a shard file."""
        return self.checkpoint_dir / shard_name

    def load_shard(self, shard_name: str) -> Dict[str, np.ndarray]:
        """Load a shard file if not already loaded.

        Args:
            shard_name: Name of the shard file

        Returns:
            Dictionary mapping parameter names to numpy arrays
        """
        if shard_name not in self._loaded_shards:
            try:
                from safetensors.numpy import load_file
            except ImportError:
                from safetensors import safe_open

                shard_path = self.get_shard_path(shard_name)
                with safe_open(str(shard_path), framework="numpy") as f:  # type: ignore[no-untyped-call]
                    self._loaded_shards[shard_name] = {
                        k: f.get_tensor(k) for k in f.keys()
                    }
                return self._loaded_shards[shard_name]

            shard_path = self.get_shard_path(shard_name)
            self._loaded_shards[shard_name] = load_file(str(shard_path))

        return self._loaded_shards[shard_name]

    def get_weight(self, name: str) -> np.ndarray:
        """Get a single weight by name.

        Args:
            name: Parameter name

        Returns:
            Numpy array of the weight

        Raises:
            KeyError: If parameter name not found
        """
        if name not in self.weight_map:
            raise KeyError(f"Parameter {name} not found in checkpoint")

        shard_name = self.weight_map[name]
        shard_data = self.load_shard(shard_name)

        return shard_data[name]

    def iter_weights(
        self,
        names: Optional[List[str]] = None,
    ) -> Iterator[Tuple[str, np.ndarray]]:
        """Iterate over weights, loading shards as needed.

        Args:
            names: Optional list of names to iterate (default: all)

        Yields:
            Tuples of (name, weight) for each parameter
        """
        if names is None:
            names = self.parameter_names

        # Group by shard for efficient loading
        shard_to_names: Dict[str, List[str]] = {}
        for name in names:
            shard = self.weight_map.get(name)
            if shard:
                shard_to_names.setdefault(shard, []).append(name)

        for shard_name, shard_names in shard_to_names.items():
            shard_data = self.load_shard(shard_name)
            for name in shard_names:
                yield name, shard_data[name]

            # Clear shard from cache after processing to save memory
            if shard_name in self._loaded_shards:
                del self._loaded_shards[shard_name]

    def clear_cache(self) -> None:
        """Clear the loaded shard cache."""
        self._loaded_shards.clear()


class WeightConverter:
    """Converts PyTorch checkpoints to MLX format.

    Handles both single-file and sharded checkpoints, performs necessary
    weight transpositions, and supports caching of converted weights.
    """

    def __init__(
        self,
        target_dtype: mx.Dtype = mx.float32,
        cache_dir: Optional[Path] = None,
        use_cache: bool = True,
    ):
        """Initialize the converter.

        Args:
            target_dtype: Target MLX dtype (default: float32 for MPS)
            cache_dir: Directory for cached weights
            use_cache: Whether to use caching
        """
        self.target_dtype = target_dtype
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.use_cache = use_cache

        # Ensure cache directory exists
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def convert_state_dict(
        self,
        state_dict: Dict[str, np.ndarray],
        validate: bool = True,
    ) -> Dict[str, mx.array]:
        """Convert a PyTorch state dict to MLX format.

        Args:
            state_dict: Dictionary of parameter name -> numpy array
            validate: Whether to validate shapes after conversion

        Returns:
            Dictionary of parameter name -> MLX array
        """
        mlx_state_dict: Dict[str, mx.array] = {}
        transpose_count = 0
        total_count = 0

        for name, weight in state_dict.items():
            total_count += 1
            original_shape = weight.shape

            # Convert the weight
            mlx_weight = convert_weight(weight, name, self.target_dtype)
            mlx_state_dict[name] = mlx_weight

            # Track transpositions
            if needs_transpose(name, original_shape):
                transpose_count += 1
                if validate:
                    # Verify transpose was applied
                    expected_shape = (original_shape[1], original_shape[0])
                    actual_shape = tuple(mlx_weight.shape)
                    if actual_shape != expected_shape:
                        raise ValueError(
                            f"Shape mismatch for {name}: "
                            f"expected {expected_shape}, got {actual_shape}"
                        )

        logger.info(
            f"Converted {total_count} weights, "
            f"transposed {transpose_count} linear weights"
        )

        return mlx_state_dict

    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
    ) -> Dict[str, np.ndarray]:
        """Load a checkpoint file or directory.

        Args:
            checkpoint_path: Path to checkpoint file or sharded directory

        Returns:
            Dictionary of parameter name -> numpy array
        """
        checkpoint_path = Path(checkpoint_path)

        if checkpoint_path.is_dir():
            # Sharded checkpoint
            return self._load_sharded_checkpoint(checkpoint_path)
        elif checkpoint_path.suffix == ".safetensors":
            return self._load_safetensors(checkpoint_path)
        elif checkpoint_path.suffix == ".pth":
            return self._load_pth(checkpoint_path)
        else:
            raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    def _load_sharded_checkpoint(
        self,
        checkpoint_dir: Path,
    ) -> Dict[str, np.ndarray]:
        """Load a sharded checkpoint directory."""
        loader = ShardedCheckpointLoader(checkpoint_dir)
        state_dict = {}

        for name, weight in loader.iter_weights():
            state_dict[name] = weight

        return state_dict

    def _load_safetensors(self, path: Path) -> Dict[str, np.ndarray]:
        """Load a single safetensors file."""
        try:
            from safetensors.numpy import load_file

            return load_file(str(path))
        except ImportError:
            from safetensors import safe_open

            with safe_open(str(path), framework="numpy") as f:  # type: ignore[no-untyped-call]
                return {k: f.get_tensor(k) for k in f.keys()}

    def _load_pth(self, path: Path) -> Dict[str, np.ndarray]:
        """Load a PyTorch .pth file."""
        import torch

        data = torch.load(str(path), map_location="cpu", weights_only=True)

        # Handle nested state_dict formats
        if isinstance(data, dict):
            if "state_dict" in data:
                data = data["state_dict"]
            elif "model" in data:
                data = data["model"]

        # Convert to numpy
        state_dict = {}
        for name, tensor in data.items():
            if isinstance(tensor, torch.Tensor):
                # Handle bfloat16
                if tensor.dtype == torch.bfloat16:
                    tensor = tensor.float()
                state_dict[name] = tensor.numpy()
            else:
                state_dict[name] = np.array(tensor)

        return state_dict

    def convert_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        model_name: Optional[str] = None,
        use_cache: Optional[bool] = None,
    ) -> Dict[str, mx.array]:
        """Convert a PyTorch checkpoint to MLX format.

        Args:
            checkpoint_path: Path to checkpoint file or directory
            model_name: Optional model name for caching
            use_cache: Override instance use_cache setting

        Returns:
            Dictionary of parameter name -> MLX array
        """
        checkpoint_path = Path(checkpoint_path)
        use_cache = use_cache if use_cache is not None else self.use_cache

        # Check cache first
        if use_cache:
            cache_path = get_cache_path(checkpoint_path, model_name, self.cache_dir)
            if cache_path.exists():
                logger.info(f"Loading cached weights from {cache_path}")
                return self._load_mlx_cache(cache_path)

        # Load and convert
        logger.info(f"Converting checkpoint from {checkpoint_path}")
        state_dict = self.load_checkpoint(checkpoint_path)
        mlx_state_dict = self.convert_state_dict(state_dict)

        # Save to cache
        if use_cache:
            self._save_mlx_cache(mlx_state_dict, cache_path)

        return mlx_state_dict

    def _load_mlx_cache(self, cache_path: Path) -> Dict[str, mx.array]:
        """Load cached MLX weights."""
        try:
            from safetensors.numpy import load_file

            np_weights = load_file(str(cache_path))
        except ImportError:
            from safetensors import safe_open

            with safe_open(str(cache_path), framework="numpy") as f:  # type: ignore[no-untyped-call]
                np_weights = {k: f.get_tensor(k) for k in f.keys()}

        # Convert to MLX arrays
        return {name: mx.array(weight) for name, weight in np_weights.items()}

    def _save_mlx_cache(
        self,
        mlx_state_dict: Dict[str, mx.array],
        cache_path: Path,
    ) -> None:
        """Save MLX weights to cache."""
        from safetensors.numpy import save_file

        # Convert MLX arrays to numpy for safetensors
        np_state_dict = {
            name: np.array(weight) for name, weight in mlx_state_dict.items()
        }

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        save_file(np_state_dict, str(cache_path))
        logger.info(f"Saved cached weights to {cache_path}")


def convert_pytorch_to_mlx(
    checkpoint_path: Union[str, Path],
    model_name: Optional[str] = None,
    target_dtype: mx.Dtype = mx.float32,
    cache_dir: Optional[Path] = None,
    use_cache: bool = True,
) -> Dict[str, mx.array]:
    """Convert PyTorch state dict to MLX format.

    This is the main entry point for weight conversion.

    Key transformations:
    1. Linear weights: transpose [out, in] -> [in, out]
    2. Convert torch.Tensor -> mx.array
    3. Handle dtype conversion
    4. Cache to ~/.cache/lingbot-world/mlx/{model_name}.safetensors

    Args:
        checkpoint_path: Path to checkpoint file or sharded directory
        model_name: Optional model name for caching
        target_dtype: Target MLX dtype (default: float32 for MPS compatibility)
        cache_dir: Optional custom cache directory
        use_cache: Whether to cache converted weights

    Returns:
        Dictionary mapping parameter names to MLX arrays

    Example:
        >>> weights = convert_pytorch_to_mlx(
        ...     'models/lingbot-world-base-cam/high_noise_model',
        ...     model_name='wanmodel_high_noise',
        ... )
        >>> print(weights['blocks.0.self_attn.q.weight'].shape)
        [2048, 2048]  # Transposed from [2048, 2048] (square) or [out, in]
    """
    converter = WeightConverter(
        target_dtype=target_dtype,
        cache_dir=cache_dir,
        use_cache=use_cache,
    )
    return converter.convert_checkpoint(checkpoint_path, model_name)


def load_mlx_weights(
    checkpoint_path: Union[str, Path],
    model_name: Optional[str] = None,
    cache_dir: Optional[Path] = None,
) -> Dict[str, mx.array]:
    """Load MLX weights from cache or convert from PyTorch.

    Convenience function that checks for cached weights first.

    Args:
        checkpoint_path: Path to checkpoint
        model_name: Optional model name
        cache_dir: Optional cache directory

    Returns:
        Dictionary of MLX weights
    """
    return convert_pytorch_to_mlx(
        checkpoint_path,
        model_name=model_name,
        cache_dir=cache_dir,
        use_cache=True,
    )


def validate_weight_shapes(
    mlx_weights: Dict[str, mx.array],
    expected_shapes: Dict[str, Tuple[int, ...]],
) -> List[str]:
    """Validate that MLX weights have expected shapes.

    Args:
        mlx_weights: Dictionary of MLX weights
        expected_shapes: Dictionary of expected shapes

    Returns:
        List of error messages (empty if all shapes match)
    """
    errors = []

    for name, expected in expected_shapes.items():
        if name not in mlx_weights:
            errors.append(f"Missing weight: {name}")
            continue

        actual = tuple(mlx_weights[name].shape)
        if actual != expected:
            errors.append(
                f"Shape mismatch for {name}: expected {expected}, got {actual}"
            )

    return errors
