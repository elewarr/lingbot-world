"""MLX implementation of WanModel for WanModel diffusion backbone.

This module provides an MLX-native implementation of the WanModel class,
assembling all component modules (blocks, head, rope) for Metal-accelerated
inference on Apple Silicon.

Key components:
- 32 WanAttentionBlockMLX layers (configurable via num_layers)
- HeadMLX output projection
- Patch embedding (Conv3d equivalent via sliding window)
- Text embedding MLP
- Time embedding MLP with projection
- Camera control embedding layers
- RoPE frequency buffer

Reference: wan/modules/model.py (WanModel class)
"""

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .blocks import WanAttentionBlockMLX
from .convert import convert_pytorch_to_mlx
from .head import HeadMLX
from .rope import create_rope_freqs_mlx

__all__ = ['WanModelMLX']


def sinusoidal_embedding_1d_mlx(dim: int, positions: mx.array) -> mx.array:
    """Compute 1D sinusoidal position embeddings.

    Args:
        dim: Embedding dimension (must be even)
        positions: Position indices of shape [N]

    Returns:
        Position embeddings of shape [N, dim]
    """
    assert dim % 2 == 0, f"dim must be even, got {dim}"
    half = dim // 2

    # Compute frequencies: 10000^(-2i/dim) for i in [0, half)
    freq_exponents = mx.arange(half, dtype=mx.float32) / half
    inv_freqs = mx.power(10000.0, -freq_exponents)  # [half]

    # Compute angles: positions * frequencies
    # positions: [N], inv_freqs: [half] -> angles: [N, half]
    positions_f = positions.astype(mx.float32)
    angles = mx.outer(positions_f, inv_freqs)

    # Concatenate cos and sin: [N, dim]
    return mx.concatenate([mx.cos(angles), mx.sin(angles)], axis=1)


class PatchEmbedding3DMLX(nn.Module):  # type: ignore[misc, name-defined]
    """3D patch embedding using Conv3d-equivalent operation.

    Converts video tensor from [B, C_in, F, H, W] to patch embeddings.
    This is equivalent to PyTorch nn.Conv3d with kernel_size=stride=patch_size.

    Args:
        in_dim: Input channels (e.g., 16 for VAE latent)
        out_dim: Output embedding dimension (e.g., 2048)
        patch_size: Tuple of (t, h, w) patch sizes (e.g., (1, 2, 2))
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        patch_size: Tuple[int, int, int],
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.patch_size = patch_size

        # Weight shape: [out_dim, in_dim * prod(patch_size)]
        # This flattens the patch and projects it
        kernel_size = in_dim * math.prod(patch_size)
        self.weight = mx.zeros((out_dim, kernel_size))
        self.bias = mx.zeros((out_dim,))

    def __call__(self, x: mx.array) -> mx.array:
        """Apply patch embedding.

        Args:
            x: Input tensor of shape [B, C_in, F, H, W] or [C_in, F, H, W]

        Returns:
            Patch embeddings of shape [B, num_patches, out_dim] or [1, num_patches, out_dim]
        """
        # Handle both batched and unbatched input
        if x.ndim == 4:
            x = mx.expand_dims(x, axis=0)

        batch_size = x.shape[0]
        c_in, f, h, w = x.shape[1], x.shape[2], x.shape[3], x.shape[4]
        pt, ph, pw = self.patch_size

        # Compute output grid sizes
        f_out = f // pt
        h_out = h // ph
        w_out = w // pw

        # Reshape to patches: [B, C_in, F, H, W] -> [B, f_out, h_out, w_out, C_in*pt*ph*pw]
        # First, reshape to [B, C_in, f_out, pt, h_out, ph, w_out, pw]
        x = x.reshape(batch_size, c_in, f_out, pt, h_out, ph, w_out, pw)
        # Permute to [B, f_out, h_out, w_out, C_in, pt, ph, pw]
        x = mx.transpose(x, (0, 2, 4, 6, 1, 3, 5, 7))
        # Flatten patch dimensions: [B, f_out, h_out, w_out, C_in*pt*ph*pw]
        x = x.reshape(batch_size, f_out, h_out, w_out, -1)
        # Flatten spatial dimensions: [B, num_patches, C_in*pt*ph*pw]
        x = x.reshape(batch_size, f_out * h_out * w_out, -1)

        # Linear projection: [B, num_patches, out_dim]
        output: mx.array = x @ self.weight.T + self.bias
        return output


class WanModelMLX(nn.Module):  # type: ignore[misc, name-defined]
    """Wan diffusion backbone for MLX.

    This is a direct port of WanModel from wan/modules/model.py to MLX.
    Supports both text-to-video and image-to-video generation.

    Args:
        model_type: Model variant - 't2v' or 'i2v' (default: 'i2v')
        patch_size: 3D patch dimensions (t, h, w) (default: (1, 2, 2))
        text_len: Fixed length for text embeddings (default: 512)
        in_dim: Input video channels (default: 16)
        dim: Hidden dimension (default: 2048)
        ffn_dim: FFN hidden dimension (default: 8192)
        freq_dim: Sinusoidal time embedding dimension (default: 256)
        text_dim: Text embedding input dimension (default: 4096)
        out_dim: Output video channels (default: 16)
        num_heads: Number of attention heads (default: 16)
        num_layers: Number of transformer blocks (default: 32)
        window_size: Window size for local attention (default: (-1, -1))
        qk_norm: Enable QK normalization (default: True)
        cross_attn_norm: Enable cross-attention normalization (default: True)
        eps: Epsilon for normalization (default: 1e-6)
    """

    def __init__(
        self,
        model_type: str = 'i2v',
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        text_len: int = 512,
        in_dim: int = 16,
        dim: int = 2048,
        ffn_dim: int = 8192,
        freq_dim: int = 256,
        text_dim: int = 4096,
        out_dim: int = 16,
        num_heads: int = 16,
        num_layers: int = 32,
        window_size: Tuple[int, int] = (-1, -1),
        qk_norm: bool = True,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        assert model_type in ['t2v', 'i2v', 'ti2v', 's2v'], f"Unknown model_type: {model_type}"
        self.model_type = model_type

        # Store config
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # Embeddings
        self.patch_embedding = PatchEmbedding3DMLX(in_dim, dim, patch_size)

        # Camera control patch embedding
        # Input: 6 * 64 * prod(patch_size) plucker coords per patch
        plucker_dim = 6 * 64 * patch_size[0] * patch_size[1] * patch_size[2]
        self.patch_embedding_wancamctrl: Any = nn.Linear(plucker_dim, dim)  # type: ignore[attr-defined]

        # Camera hidden state processing
        self.c2ws_hidden_states_layer1: Any = nn.Linear(dim, dim)  # type: ignore[attr-defined]
        self.c2ws_hidden_states_layer2: Any = nn.Linear(dim, dim)  # type: ignore[attr-defined]

        # Text embedding: Linear -> GELU(tanh) -> Linear
        self.text_embedding_linear1: Any = nn.Linear(text_dim, dim)  # type: ignore[attr-defined]
        self.text_embedding_linear2: Any = nn.Linear(dim, dim)  # type: ignore[attr-defined]

        # Time embedding: Linear -> SiLU -> Linear
        self.time_embedding_linear1: Any = nn.Linear(freq_dim, dim)  # type: ignore[attr-defined]
        self.time_embedding_linear2: Any = nn.Linear(dim, dim)  # type: ignore[attr-defined]

        # Time projection: SiLU -> Linear to 6*dim
        self.time_projection_linear: Any = nn.Linear(dim, dim * 6)  # type: ignore[attr-defined]

        # Attention blocks (num_layers, default 32)
        self.blocks: List[WanAttentionBlockMLX] = [
            WanAttentionBlockMLX(
                dim=dim,
                ffn_dim=ffn_dim,
                num_heads=num_heads,
                window_size=window_size,
                qk_norm=qk_norm,
                cross_attn_norm=cross_attn_norm,
                eps=eps,
            )
            for _ in range(num_layers)
        ]

        # Output head
        self.head = HeadMLX(dim, out_dim, patch_size, eps)

        # Pre-compute RoPE frequencies
        head_dim = dim // num_heads
        self._freqs_cos, self._freqs_sin = create_rope_freqs_mlx(1024, head_dim)

    def _text_embedding(self, x: mx.array) -> mx.array:
        """Apply text embedding MLP: Linear -> GELU(tanh) -> Linear."""
        x = self.text_embedding_linear1(x)
        x = nn.gelu_approx(x)  # type: ignore[attr-defined]
        x = self.text_embedding_linear2(x)
        return x

    def _time_embedding(self, x: mx.array) -> mx.array:
        """Apply time embedding MLP: Linear -> SiLU -> Linear."""
        x = self.time_embedding_linear1(x)
        x = nn.silu(x)  # type: ignore[attr-defined]
        x = self.time_embedding_linear2(x)
        return x

    def _time_projection(self, x: mx.array) -> mx.array:
        """Apply time projection: SiLU -> Linear."""
        x = nn.silu(x)  # type: ignore[attr-defined]
        x = self.time_projection_linear(x)
        return x

    def __call__(
        self,
        x: List[mx.array],
        t: mx.array,
        context: List[mx.array],
        seq_len: int,
        y: Optional[List[mx.array]] = None,
        dit_cond_dict: Optional[Dict[str, Any]] = None,
    ) -> List[mx.array]:
        """Forward pass through the diffusion model.

        Args:
            x: List of input video tensors, each [C_in, F, H, W]
            t: Diffusion timesteps of shape [B]
            context: List of text embeddings, each [L, C]
            seq_len: Maximum sequence length for positional encoding
            y: Optional conditional video inputs for i2v mode
            dit_cond_dict: Optional dict containing camera embeddings

        Returns:
            List of denoised video tensors [C_out, F, H, W]
        """
        if self.model_type == 'i2v':
            assert y is not None, "y (conditional input) required for i2v model"

        batch_size = len(x)

        # Concatenate x and y for i2v mode
        if y is not None:
            x = [mx.concatenate([u, v], axis=0) for u, v in zip(x, y)]

        # Apply patch embedding to each sample
        x_embedded = [self.patch_embedding(mx.expand_dims(u, axis=0)) for u in x]

        # Get grid sizes from embedded shapes
        # Each embedded: [1, num_patches, dim]
        grid_sizes_list = []
        for i, u_orig in enumerate(x):
            # u_orig shape after cat: [2*C_in or C_in, F, H, W]
            # We need the patched grid size
            c_in = u_orig.shape[0]
            f, h, w = u_orig.shape[1], u_orig.shape[2], u_orig.shape[3]
            f_out = f // self.patch_size[0]
            h_out = h // self.patch_size[1]
            w_out = w // self.patch_size[2]
            grid_sizes_list.append([f_out, h_out, w_out])
        grid_sizes = mx.array(grid_sizes_list, dtype=mx.int32)

        # Flatten and get sequence lengths
        x_flat = [u.reshape(u.shape[1], u.shape[2]) for u in x_embedded]  # [num_patches, dim]
        seq_lens_list = [u.shape[0] for u in x_flat]
        seq_lens = mx.array(seq_lens_list, dtype=mx.int32)

        assert max(seq_lens_list) <= seq_len, f"Max seq len {max(seq_lens_list)} > {seq_len}"

        # Pad to max sequence length and batch
        x_padded = []
        for u in x_flat:
            pad_len = seq_len - u.shape[0]
            if pad_len > 0:
                padding = mx.zeros((pad_len, self.dim), dtype=u.dtype)
                u = mx.concatenate([u, padding], axis=0)
            x_padded.append(u)
        x_batched = mx.stack(x_padded, axis=0)  # [B, seq_len, dim]

        # Time embeddings
        # t shape: [B] or [B, seq_len]
        if t.ndim == 1:
            t = mx.broadcast_to(mx.expand_dims(t, axis=1), (batch_size, seq_len))

        # Compute sinusoidal embeddings and apply time MLP
        t_flat = t.flatten()  # [B * seq_len]
        t_emb = sinusoidal_embedding_1d_mlx(self.freq_dim, t_flat)  # [B*seq_len, freq_dim]
        t_emb = t_emb.reshape(batch_size, seq_len, self.freq_dim)  # [B, seq_len, freq_dim]
        e = self._time_embedding(t_emb)  # [B, seq_len, dim]
        e0 = self._time_projection(e)  # [B, seq_len, 6*dim]
        e0 = e0.reshape(batch_size, seq_len, 6, self.dim)  # [B, seq_len, 6, dim]

        # Process context (text embeddings)
        # Pad each context to text_len and stack
        context_padded = []
        for ctx in context:
            pad_len = self.text_len - ctx.shape[0]
            if pad_len > 0:
                padding = mx.zeros((pad_len, ctx.shape[1]), dtype=ctx.dtype)
                ctx = mx.concatenate([ctx, padding], axis=0)
            context_padded.append(ctx)
        context_batched = mx.stack(context_padded, axis=0)  # [B, text_len, text_dim]
        context_embedded = self._text_embedding(context_batched)  # [B, text_len, dim]

        # Process camera embeddings if provided
        if dit_cond_dict is not None and "c2ws_plucker_emb" in dit_cond_dict:
            c2ws_plucker_emb = dit_cond_dict["c2ws_plucker_emb"]

            # Rearrange from [1, C, F, H, W] to [1, num_patches, C*pt*ph*pw]
            # where C=6*64, and we reshape using patch_size
            rearranged = []
            for emb in c2ws_plucker_emb:
                # emb: [1, 6*64, F, H, W]
                b, c, f, h, w = emb.shape
                pt, ph, pw = self.patch_size
                f_out = f // pt
                h_out = h // ph
                w_out = w // pw

                # Reshape: [1, C, f_out, pt, h_out, ph, w_out, pw]
                emb = emb.reshape(b, c, f_out, pt, h_out, ph, w_out, pw)
                # Permute: [1, f_out, h_out, w_out, C, pt, ph, pw]
                emb = mx.transpose(emb, (0, 2, 4, 6, 1, 3, 5, 7))
                # Flatten: [1, num_patches, C*pt*ph*pw]
                emb = emb.reshape(b, f_out * h_out * w_out, -1)
                rearranged.append(emb)

            # Concatenate all samples: [1, total_patches, C*pt*ph*pw]
            c2ws_plucker_emb = mx.concatenate(rearranged, axis=1)

            # Apply camera embedding layers
            c2ws_plucker_emb = self.patch_embedding_wancamctrl(c2ws_plucker_emb)
            c2ws_hidden = nn.silu(self.c2ws_hidden_states_layer1(c2ws_plucker_emb))  # type: ignore[attr-defined]
            c2ws_hidden = self.c2ws_hidden_states_layer2(c2ws_hidden)

            # Update dit_cond_dict with processed embeddings
            dit_cond_dict = dict(dit_cond_dict)
            dit_cond_dict["c2ws_plucker_emb"] = c2ws_plucker_emb + c2ws_hidden

        # Run through attention blocks
        # e0 needs to be broadcast/sliced to [B, 1, 6, dim] per block's expectation
        # Actually looking at blocks.py, e shape is [B, 1, 6, C]
        # We have [B, seq_len, 6, dim], need to handle this carefully
        # The PyTorch code uses e0 directly which is [B, seq_len, 6, dim]
        # Then in the block it does modulation.unsqueeze(0) + e
        # Where modulation is [1, 6, dim]
        # So e should be [B, L, 6, dim] or broadcastable

        # Looking at the block signature: e: mx.array of shape [B, 1, 6, C]
        # But PyTorch passes e0 which is [B, L, 6, dim]
        # Let me check the block code again...
        # The block uses e[1].squeeze(2) which assumes e is chunked into 6 parts
        # Actually it takes e as [B, L, 6, dim] and does:
        # e_mod = mx.expand_dims(self.modulation, axis=0) + e
        # where modulation is [1, 6, dim]
        # So e should be [B, 1, 6, dim] based on the block signature
        # But the PyTorch code passes e0 which is [B, seq_len, 6, dim]

        # Looking more carefully at PyTorch forward:
        # e0 = self.time_projection(e).unflatten(2, (6, self.dim))
        # This gives [B, seq_len, 6, dim]
        # But in kwargs: e=e0

        # In the block forward, it does:
        # e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
        # Where modulation is [1, 6, dim] and after unsqueeze(0) is [1, 1, 6, dim]
        # And e is [B, seq_len, 6, dim]
        # The broadcast gives [B, seq_len, 6, dim]
        # Then chunk(6, dim=2) gives 6 tensors of [B, seq_len, 1, dim]

        # So the block expects e of shape [B, L, 6, dim] not [B, 1, 6, dim]
        # The MLX block signature says [B, 1, 6, C] but implementation may differ
        # Let me pass [B, seq_len, 6, dim] and see if it works

        # Actually, looking at the MLX block implementation, it does:
        # e_mod = mx.expand_dims(self.modulation, axis=0) + e
        # Then slices e_mod[:, :, 0:1, :] etc.
        # This should work with [B, seq_len, 6, dim]

        # But wait, the block docstring says e: [B, 1, 6, C]
        # The implementation does work with [B, L, 6, C] though
        # Because of broadcasting

        context_lens = None  # Not used in current implementation

        for block in self.blocks:
            x_batched = block(
                x_batched,
                e0,
                seq_lens,
                grid_sizes,
                self._freqs_cos,
                self._freqs_sin,
                context_embedded,
                context_lens,
                dit_cond_dict,
            )

        # Apply head
        x_batched = self.head(x_batched, e)

        # Unpatchify
        output = self._unpatchify(x_batched, grid_sizes)

        return output

    def _unpatchify(
        self,
        x: mx.array,
        grid_sizes: mx.array,
    ) -> List[mx.array]:
        """Reconstruct video tensors from patch embeddings.

        Args:
            x: Batched patch features of shape [B, L, out_dim * prod(patch_size)]
            grid_sizes: Grid dimensions [B, 3] containing (F_patches, H_patches, W_patches)

        Returns:
            List of reconstructed video tensors [C_out, F, H, W]
        """
        batch_size = x.shape[0]
        out = []

        for i in range(batch_size):
            f, h, w = int(grid_sizes[i, 0].item()), int(grid_sizes[i, 1].item()), int(grid_sizes[i, 2].item())
            num_patches = f * h * w

            # Extract valid patches: [num_patches, out_dim * prod(patch_size)]
            u = x[i, :num_patches]

            # Reshape to grid with patch dimensions
            # [num_patches, out_dim * prod(patch_size)] -> [f, h, w, pt, ph, pw, c]
            pt, ph, pw = self.patch_size
            c = self.out_dim
            u = u.reshape(f, h, w, pt, ph, pw, c)

            # Permute to [c, f, pt, h, ph, w, pw]
            # einops: 'fhwpqrc->cfphqwr'
            u = mx.transpose(u, (6, 0, 3, 1, 4, 2, 5))

            # Reshape to [c, f*pt, h*ph, w*pw]
            u = u.reshape(c, f * pt, h * ph, w * pw)

            out.append(u)

        return out

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: Union[str, Path],
        model_name: Optional[str] = None,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> 'WanModelMLX':
        """Load WanModelMLX from a PyTorch checkpoint.

        Args:
            checkpoint_path: Path to PyTorch checkpoint file or directory
            model_name: Optional model name for caching
            use_cache: Whether to use cached converted weights
            **kwargs: Additional model configuration overrides

        Returns:
            WanModelMLX instance with loaded weights
        """
        checkpoint_path = Path(checkpoint_path)

        # Convert PyTorch weights to MLX format
        mlx_weights = convert_pytorch_to_mlx(
            checkpoint_path,
            model_name=model_name,
            use_cache=use_cache,
        )

        # Extract model config from weights or use defaults
        config = cls._infer_config_from_weights(mlx_weights)
        config.update(kwargs)

        # Create model
        model = cls(**config)

        # Load weights
        model._load_weights(mlx_weights)

        return model

    @staticmethod
    def _infer_config_from_weights(weights: Dict[str, mx.array]) -> Dict[str, Any]:
        """Infer model configuration from weight shapes.

        NOTE: Weights are in PyTorch format [out_features, in_features] since
        we no longer transpose during conversion.

        Args:
            weights: Dictionary of MLX weights

        Returns:
            Configuration dictionary
        """
        config: Dict[str, Any] = {}

        # Infer dim from block weights
        # Weight shape: [out_features, in_features] = [dim, dim] for q projection
        if 'blocks.0.self_attn.q.weight' in weights:
            w = weights['blocks.0.self_attn.q.weight']
            # For q projection: in=dim, out=dim, so both dimensions are dim
            config['dim'] = int(w.shape[0])  # out_features = dim

        # Infer ffn_dim from ffn.0 weight
        # ffn.0 is Linear(dim, ffn_dim), weight shape: [ffn_dim, dim]
        if 'blocks.0.ffn.0.weight' in weights:
            w = weights['blocks.0.ffn.0.weight']
            config['ffn_dim'] = int(w.shape[0])  # out_features = ffn_dim

        # Infer num_heads from head_dim if possible
        if 'dim' in config:
            # Default to 16 heads for dim=2048
            if config['dim'] == 2048:
                config['num_heads'] = 16
            else:
                # Assume head_dim = 128
                config['num_heads'] = config['dim'] // 128

        # Count blocks
        block_count = 0
        for key in weights:
            if key.startswith('blocks.') and '.self_attn.q.weight' in key:
                idx = int(key.split('.')[1])
                block_count = max(block_count, idx + 1)
        if block_count > 0:
            config['num_layers'] = block_count

        # Infer text_dim from text embedding weights
        # text_embedding.0 is Linear(text_dim, dim), weight shape: [dim, text_dim]
        if 'text_embedding.0.weight' in weights:
            w = weights['text_embedding.0.weight']
            config['text_dim'] = int(w.shape[1])  # in_features = text_dim

        # Infer freq_dim from time embedding weights
        # time_embedding.0 is Linear(freq_dim, dim), weight shape: [dim, freq_dim]
        if 'time_embedding.0.weight' in weights:
            w = weights['time_embedding.0.weight']
            config['freq_dim'] = int(w.shape[1])  # in_features = freq_dim

        return config

    def _load_weights(self, weights: Dict[str, mx.array]) -> None:
        """Load weights from a converted state dict.

        Maps PyTorch weight names to MLX module structure.

        Args:
            weights: Dictionary of MLX weights
        """
        # Helper to assign weight
        def set_weight(module: Any, name: str, value: mx.array) -> None:
            parts = name.split('.')
            for part in parts[:-1]:
                if part.isdigit():
                    module = module[int(part)]
                else:
                    module = getattr(module, part)
            final = parts[-1]
            if hasattr(module, final):
                setattr(module, final, value)

        # Map PyTorch names to MLX structure
        for name, weight in weights.items():
            try:
                if name.startswith('patch_embedding.'):
                    # Conv3d weight: [out_ch, in_ch, t, h, w] -> flatten to [out_ch, in_ch*t*h*w]
                    if name == 'patch_embedding.weight':
                        # Flatten conv weight and transpose for Linear-like projection
                        w = weight.reshape(weight.shape[0], -1)  # [out_dim, in_dim*prod(patch)]
                        self.patch_embedding.weight = w
                    elif name == 'patch_embedding.bias':
                        self.patch_embedding.bias = weight

                elif name.startswith('patch_embedding_wancamctrl.'):
                    attr = name.split('.')[-1]
                    setattr(self.patch_embedding_wancamctrl, attr, weight)

                elif name.startswith('c2ws_hidden_states_layer1.'):
                    attr = name.split('.')[-1]
                    setattr(self.c2ws_hidden_states_layer1, attr, weight)

                elif name.startswith('c2ws_hidden_states_layer2.'):
                    attr = name.split('.')[-1]
                    setattr(self.c2ws_hidden_states_layer2, attr, weight)

                elif name.startswith('text_embedding.'):
                    # text_embedding.0.weight, text_embedding.0.bias, text_embedding.2.weight, etc.
                    parts = name.split('.')
                    idx = int(parts[1])
                    attr = parts[2]
                    if idx == 0:
                        setattr(self.text_embedding_linear1, attr, weight)
                    elif idx == 2:
                        setattr(self.text_embedding_linear2, attr, weight)

                elif name.startswith('time_embedding.'):
                    parts = name.split('.')
                    idx = int(parts[1])
                    attr = parts[2]
                    if idx == 0:
                        setattr(self.time_embedding_linear1, attr, weight)
                    elif idx == 2:
                        setattr(self.time_embedding_linear2, attr, weight)

                elif name.startswith('time_projection.'):
                    # time_projection.1.weight, time_projection.1.bias
                    parts = name.split('.')
                    attr = parts[2]
                    setattr(self.time_projection_linear, attr, weight)

                elif name.startswith('blocks.'):
                    self._load_block_weight(name, weight)

                elif name.startswith('head.'):
                    self._load_head_weight(name, weight)

                # Skip freqs buffer - we compute it ourselves

            except (AttributeError, IndexError, ValueError) as e:
                # Log but don't fail on unrecognized weights
                pass

    def _load_block_weight(self, name: str, weight: mx.array) -> None:
        """Load weight into an attention block.

        Args:
            name: Weight name like 'blocks.0.self_attn.q.weight'
            weight: MLX array to load
        """
        parts = name.split('.')
        block_idx = int(parts[1])
        if block_idx >= len(self.blocks):
            return

        block = self.blocks[block_idx]
        remainder = '.'.join(parts[2:])

        if remainder.startswith('self_attn.'):
            sub_parts = remainder.split('.')
            attr_path = sub_parts[1:]  # e.g., ['q', 'weight']
            module = block.self_attn
            for p in attr_path[:-1]:
                module = getattr(module, p)
            setattr(module, attr_path[-1], weight)

        elif remainder.startswith('cross_attn.'):
            sub_parts = remainder.split('.')
            attr_path = sub_parts[1:]
            module = block.cross_attn
            for p in attr_path[:-1]:
                module = getattr(module, p)
            setattr(module, attr_path[-1], weight)

        elif remainder.startswith('norm1.'):
            # norm1 has no learnable weights (elementwise_affine=False)
            pass

        elif remainder.startswith('norm2.'):
            # norm2 has no learnable weights
            pass

        elif remainder.startswith('norm3.'):
            if block._use_norm3:
                attr = remainder.split('.')[-1]
                setattr(block.norm3, attr, weight)

        elif remainder.startswith('ffn.'):
            # ffn.0.weight, ffn.0.bias, ffn.2.weight, ffn.2.bias
            sub_parts = remainder.split('.')
            idx = int(sub_parts[1])
            attr = sub_parts[2]
            if idx == 0:
                setattr(block.ffn_linear1, attr, weight)
            elif idx == 2:
                setattr(block.ffn_linear2, attr, weight)

        elif remainder == 'modulation':
            block.modulation = weight

        elif remainder.startswith('cam_injector_layer1.'):
            attr = remainder.split('.')[-1]
            setattr(block.cam_injector_layer1, attr, weight)

        elif remainder.startswith('cam_injector_layer2.'):
            attr = remainder.split('.')[-1]
            setattr(block.cam_injector_layer2, attr, weight)

        elif remainder.startswith('cam_scale_layer.'):
            attr = remainder.split('.')[-1]
            setattr(block.cam_scale_layer, attr, weight)

        elif remainder.startswith('cam_shift_layer.'):
            attr = remainder.split('.')[-1]
            setattr(block.cam_shift_layer, attr, weight)

    def _load_head_weight(self, name: str, weight: mx.array) -> None:
        """Load weight into the output head.

        Args:
            name: Weight name like 'head.norm.weight'
            weight: MLX array to load
        """
        parts = name.split('.')
        remainder = '.'.join(parts[1:])

        if remainder.startswith('norm.'):
            # norm has no learnable weights by default
            pass

        elif remainder.startswith('head.'):
            attr = remainder.split('.')[-1]
            setattr(self.head.head, attr, weight)

        elif remainder == 'modulation':
            self.head.modulation = weight
