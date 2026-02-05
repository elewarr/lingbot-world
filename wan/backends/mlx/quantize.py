"""Quantization utilities for WanModelMLX.

This module provides quantization support for the MLX backend, enabling
reduced memory usage and faster inference on Apple Silicon.

Supported quantization modes:
- int8: 8-bit affine quantization (~70% memory reduction)
- int4: 4-bit affine quantization (~84% memory reduction)
- nf4: NormalFloat4 via nvfp4 mode (~85% memory reduction)

Example usage:
    from wan.backends.mlx import WanModelMLX
    from wan.backends.mlx.quantize import quantize_model

    model = WanModelMLX.from_pretrained("path/to/checkpoint")
    quantized_model = quantize_model(model, quant_type='int4')
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

__all__ = [
    'quantize_model',
    'save_quantized_model',
    'load_quantized_model',
    'get_model_memory_bytes',
    'QuantizationConfig',
    'DEFAULT_EXCLUDE_PATTERNS',
]

logger = logging.getLogger(__name__)

# Default layer patterns to exclude from quantization
# These layers are critical for model quality and should remain in full precision
DEFAULT_EXCLUDE_PATTERNS = [
    'patch_embedding',           # Input patch embedding (critical for quality)
    'head.head',                 # Output head (critical for quality)
    'patch_embedding_wancamctrl',  # Camera control embedding
]

# Quantization type to MLX mode/bits mapping
QUANT_TYPE_MAP: Dict[str, Dict[str, Any]] = {
    'int8': {'mode': 'affine', 'bits': 8, 'group_size': 64},
    'int4': {'mode': 'affine', 'bits': 4, 'group_size': 64},
    'nf4': {'mode': 'nvfp4', 'bits': None, 'group_size': None},  # nvfp4 has fixed params
    'mxfp4': {'mode': 'mxfp4', 'bits': None, 'group_size': None},
}


class QuantizationConfig:
    """Configuration for model quantization.

    Attributes:
        quant_type: Quantization type ('int8', 'int4', 'nf4', 'mxfp4')
        group_size: Group size for quantization (None uses defaults)
        bits: Bit width (None uses defaults for the mode)
        exclude_layers: Layer name patterns to exclude from quantization
    """

    def __init__(
        self,
        quant_type: Literal['int8', 'int4', 'nf4', 'mxfp4'] = 'int4',
        group_size: Optional[int] = None,
        bits: Optional[int] = None,
        exclude_layers: Optional[List[str]] = None,
    ) -> None:
        if quant_type not in QUANT_TYPE_MAP:
            raise ValueError(
                f"Unknown quant_type: {quant_type}. "
                f"Supported: {list(QUANT_TYPE_MAP.keys())}"
            )

        self.quant_type = quant_type
        # Use explicit None check to allow empty list
        if exclude_layers is None:
            self.exclude_layers = list(DEFAULT_EXCLUDE_PATTERNS)
        else:
            self.exclude_layers = exclude_layers

        # Get default params for this quant type
        defaults = QUANT_TYPE_MAP[quant_type]
        self.mode = defaults['mode']
        self.group_size = group_size or defaults['group_size']
        self.bits = bits or defaults['bits']

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            'quant_type': self.quant_type,
            'mode': self.mode,
            'group_size': self.group_size,
            'bits': self.bits,
            'exclude_layers': self.exclude_layers,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'QuantizationConfig':
        """Create config from dictionary."""
        return cls(
            quant_type=d.get('quant_type', 'int4'),
            group_size=d.get('group_size'),
            bits=d.get('bits'),
            exclude_layers=d.get('exclude_layers'),
        )


def get_model_memory_bytes(model: Any) -> int:
    """Calculate total memory used by model parameters.

    Args:
        model: MLX model

    Returns:
        Total memory in bytes
    """
    params = model.parameters()
    leaves = tree_flatten(params)
    return sum(v.nbytes for k, v in leaves)  # type: ignore[misc]


def _should_quantize(path: str, module: Any, exclude_patterns: List[str]) -> bool:
    """Determine if a module should be quantized.

    Args:
        path: Module path in the model (e.g., 'blocks.0.self_attn.q')
        module: The module instance
        exclude_patterns: List of patterns to exclude

    Returns:
        True if module should be quantized
    """
    # Only quantize modules that have to_quantized method (Linear, Embedding, etc.)
    if not hasattr(module, 'to_quantized'):
        return False
    
    # Check if path matches any exclude pattern
    for pattern in exclude_patterns:
        if pattern in path:
            return False
    return True


def quantize_model(
    model: Any,
    quant_type: Literal['int8', 'int4', 'nf4', 'mxfp4'] = 'int4',
    exclude_layers: Optional[List[str]] = None,
    group_size: Optional[int] = None,
    bits: Optional[int] = None,
) -> Any:
    """Quantize model weights for reduced memory usage.

    This function applies quantization to Linear layers in the model,
    converting them to QuantizedLinear. Specified layers can be excluded
    to preserve quality-critical components.

    Args:
        model: WanModelMLX or any MLX nn.Module
        quant_type: Quantization type
            - 'int8': 8-bit affine quantization (~70% memory reduction)
            - 'int4': 4-bit affine quantization (~84% memory reduction)
            - 'nf4': NormalFloat4 via nvfp4 (~85% memory reduction)
            - 'mxfp4': MX FP4 format (~86% memory reduction)
        exclude_layers: Layer name patterns to exclude (default: patch_embedding, head)
        group_size: Override group size (default depends on quant_type)
        bits: Override bit width (default depends on quant_type)

    Returns:
        The same model instance with quantized layers (modified in-place)

    Example:
        >>> model = WanModelMLX(...)
        >>> model = quantize_model(model, quant_type='int4')
        >>> # Model now uses ~84% less memory
    """
    config = QuantizationConfig(
        quant_type=quant_type,
        group_size=group_size,
        bits=bits,
        exclude_layers=exclude_layers,
    )

    # Get memory before quantization
    mem_before = get_model_memory_bytes(model)

    # Create predicate for selective quantization
    def class_predicate(path: str, module: Any) -> bool:
        return _should_quantize(path, module, config.exclude_layers)

    # Build kwargs for nn.quantize
    quant_kwargs: Dict[str, Any] = {'mode': config.mode}
    if config.group_size is not None:
        quant_kwargs['group_size'] = config.group_size
    if config.bits is not None:
        quant_kwargs['bits'] = config.bits
    quant_kwargs['class_predicate'] = class_predicate

    # Apply quantization in-place
    nn.quantize(model, **quant_kwargs)  # type: ignore[attr-defined]

    # Evaluate to materialize quantized weights
    mx.eval(model.parameters())

    # Get memory after quantization
    mem_after = get_model_memory_bytes(model)
    reduction = (1 - mem_after / mem_before) * 100 if mem_before > 0 else 0

    logger.info(
        f"Quantized model with {quant_type}: "
        f"{mem_before / 1024**2:.1f} MB -> {mem_after / 1024**2:.1f} MB "
        f"({reduction:.1f}% reduction)"
    )

    # Store config on model for later saving
    setattr(model, '_quantization_config', config)

    return model


def save_quantized_model(
    model: Any,
    path: Union[str, Path],
    save_config: bool = True,
) -> None:
    """Save quantized model weights and config.

    Saves weights to safetensors format and optionally saves
    quantization config as JSON for later loading.

    Args:
        model: Quantized MLX model
        path: Output path (directory or .safetensors file)
        save_config: Whether to save quantization config JSON

    Example:
        >>> model = quantize_model(model, 'int4')
        >>> save_quantized_model(model, 'quantized_model/')
    """
    path = Path(path)

    # Determine output paths
    if path.suffix == '.safetensors':
        weights_path = path
        config_path = path.with_suffix('.json')
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        path.mkdir(parents=True, exist_ok=True)
        weights_path = path / 'weights.safetensors'
        config_path = path / 'quantization_config.json'

    # Save weights
    params = model.parameters()
    flat = tree_flatten(params)
    weights_dict = dict(flat)
    mx.save_safetensors(str(weights_path), weights_dict)
    logger.info(f"Saved {len(weights_dict)} parameters to {weights_path}")

    # Save config if available and requested
    if save_config and hasattr(model, '_quantization_config'):
        config: QuantizationConfig = model._quantization_config
        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        logger.info(f"Saved quantization config to {config_path}")


def load_quantized_model(
    path: Union[str, Path],
    model: Optional[Any] = None,
    model_cls: Optional[type] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> Any:
    """Load a pre-quantized model from disk.

    Args:
        path: Path to saved quantized model (directory or .safetensors file)
        model: Optional existing model instance to load weights into.
            If provided, model will be quantized to match saved weights.
        model_cls: Model class to instantiate (e.g., WanModelMLX).
            Required if model is not provided.
        model_kwargs: Keyword arguments for model instantiation.

    Returns:
        Model with loaded quantized weights

    Example:
        >>> from wan.backends.mlx import WanModelMLX
        >>> model = load_quantized_model('quantized_model/', model_cls=WanModelMLX)
    """
    path = Path(path)

    # Determine paths
    if path.suffix == '.safetensors':
        weights_path = path
        config_path = path.with_suffix('.json')
    else:
        weights_path = path / 'weights.safetensors'
        config_path = path / 'quantization_config.json'

    # Load config if available
    config: Optional[QuantizationConfig] = None
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = QuantizationConfig.from_dict(json.load(f))

    # Create or prepare model
    if model is None:
        if model_cls is None:
            raise ValueError("Either model or model_cls must be provided")
        model_kwargs = model_kwargs or {}
        model = model_cls(**model_kwargs)
        mx.eval(model.parameters())

    # Apply quantization structure to match saved weights
    if config is not None:
        quant_kwargs: Dict[str, Any] = {'mode': config.mode}
        if config.group_size is not None:
            quant_kwargs['group_size'] = config.group_size
        if config.bits is not None:
            quant_kwargs['bits'] = config.bits

        def class_predicate(p: str, m: Any) -> bool:
            return _should_quantize(p, m, config.exclude_layers)

        quant_kwargs['class_predicate'] = class_predicate
        nn.quantize(model, **quant_kwargs)  # type: ignore[attr-defined]

    # Load weights
    loaded_weights = mx.load(str(weights_path))
    if isinstance(loaded_weights, dict):
        model.load_weights(list(loaded_weights.items()))
    else:
        # Handle tuple return type from mx.load
        weights_dict = loaded_weights[0] if isinstance(loaded_weights, tuple) else loaded_weights
        if isinstance(weights_dict, dict):
            model.load_weights(list(weights_dict.items()))
    mx.eval(model.parameters())

    # Store config on model
    if config is not None:
        setattr(model, '_quantization_config', config)

    logger.info(f"Loaded quantized model from {path}")
    return model
