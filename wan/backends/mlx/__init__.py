# MLX Backend for WanModel
# Provides native Metal-accelerated operations on Apple Silicon

from .norms import WanRMSNormMLX, WanLayerNormMLX
from .rope import rope_params_mlx, rope_apply_mlx, create_rope_freqs_mlx
from .attention import WanSelfAttentionMLX, WanCrossAttentionMLX
from .blocks import WanAttentionBlockMLX
from .head import HeadMLX
from .model import WanModelMLX
from .convert import (
    convert_pytorch_to_mlx,
    load_mlx_weights,
    get_cache_path,
    is_linear_weight,
    needs_transpose,
    convert_weight,
    WeightConverter,
    ShardedCheckpointLoader,
)
from .quantize import (
    quantize_model,
    save_quantized_model,
    load_quantized_model,
    get_model_memory_bytes,
    QuantizationConfig,
    DEFAULT_EXCLUDE_PATTERNS,
)
from .backend import MLXBackend

__all__ = [
    # Norm layers
    'WanRMSNormMLX',
    'WanLayerNormMLX',
    # RoPE
    'rope_params_mlx',
    'rope_apply_mlx',
    'create_rope_freqs_mlx',
    # Attention
    'WanSelfAttentionMLX',
    'WanCrossAttentionMLX',
    # Blocks
    'WanAttentionBlockMLX',
    # Head
    'HeadMLX',
    # Model
    'WanModelMLX',
    # Weight conversion
    'convert_pytorch_to_mlx',
    'load_mlx_weights',
    'get_cache_path',
    'is_linear_weight',
    'needs_transpose',
    'convert_weight',
    'WeightConverter',
    'ShardedCheckpointLoader',
    # Quantization
    'quantize_model',
    'save_quantized_model',
    'load_quantized_model',
    'get_model_memory_bytes',
    'QuantizationConfig',
    'DEFAULT_EXCLUDE_PATTERNS',
    # Backend
    'MLXBackend',
]
