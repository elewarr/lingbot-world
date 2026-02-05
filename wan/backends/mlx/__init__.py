# MLX Backend for WanModel
# Provides native Metal-accelerated operations on Apple Silicon

from .norms import WanRMSNormMLX, WanLayerNormMLX
from .rope import rope_params_mlx, rope_apply_mlx, create_rope_freqs_mlx
from .attention import WanSelfAttentionMLX, WanCrossAttentionMLX
from .blocks import WanAttentionBlockMLX
from .head import HeadMLX
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
    # Weight conversion
    'convert_pytorch_to_mlx',
    'load_mlx_weights',
    'get_cache_path',
    'is_linear_weight',
    'needs_transpose',
    'convert_weight',
    'WeightConverter',
    'ShardedCheckpointLoader',
]
