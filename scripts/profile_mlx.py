#!/usr/bin/env python3
"""MLX Forward Pass Profiler - Identify bottlenecks in WanModelMLX.

This script profiles each component of the MLX forward pass to identify
where optimization efforts should be focused.

Components profiled:
1. Embeddings (patch, text, time, camera)
2. Per-block breakdown (self-attn, cross-attn, FFN, norms, camera injection)
3. RoPE computation
4. Output head

Usage:
    source .venv/bin/activate
    
    # Quick test with synthetic model
    python scripts/profile_mlx.py --mode synthetic --blocks 4
    
    # Full model profiling
    python scripts/profile_mlx.py --ckpt_dir models/lingbot-world-base-cam
    
    # Profile specific number of blocks
    python scripts/profile_mlx.py --blocks 8 --frames 11 --steps 3
"""

import argparse
import gc
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# =============================================================================
# Type Definitions
# =============================================================================


@dataclass
class ComponentTiming:
    """Timing data for a single component."""
    name: str
    times: List[float] = field(default_factory=list)
    
    @property
    def mean_ms(self) -> float:
        return float(np.mean(self.times)) * 1000 if self.times else 0.0
    
    @property
    def std_ms(self) -> float:
        return float(np.std(self.times)) * 1000 if len(self.times) > 1 else 0.0
    
    @property
    def total_ms(self) -> float:
        return sum(self.times) * 1000


@dataclass
class ProfileResult:
    """Complete profiling result."""
    timings: Dict[str, ComponentTiming] = field(default_factory=dict)
    total_time_sec: float = 0.0
    num_runs: int = 0
    model_info: str = ""
    seq_len: int = 0
    num_blocks: int = 0
    
    def add_timing(self, name: str, elapsed: float) -> None:
        if name not in self.timings:
            self.timings[name] = ComponentTiming(name=name)
        self.timings[name].times.append(elapsed)


# =============================================================================
# Profiling Implementation
# =============================================================================

def profile_mlx_forward(
    mode: str = "synthetic",
    ckpt_dir: Optional[str] = None,
    num_blocks: int = 4,
    frames: int = 11,
    num_runs: int = 3,
    warmup_runs: int = 1,
) -> ProfileResult:
    """Profile the MLX forward pass component by component.
    
    Args:
        mode: 'synthetic' for small test model, 'full' for real weights
        ckpt_dir: Path to model checkpoint (for 'full' mode)
        num_blocks: Number of blocks to profile (for 'synthetic' mode)
        frames: Number of video frames (affects sequence length)
        num_runs: Number of profiling runs
        warmup_runs: Number of warmup runs before timing
    
    Returns:
        ProfileResult with per-component timings
    """
    import mlx.core as mx
    import mlx.nn as nn
    
    result = ProfileResult()
    result.num_runs = num_runs
    
    if mode == "synthetic":
        model, model_config = _create_synthetic_model(num_blocks)
        result.model_info = f"Synthetic ({num_blocks} blocks, dim={model_config['dim']})"
        result.num_blocks = num_blocks
    else:
        model = _load_full_model(ckpt_dir)
        result.model_info = f"Full model from {ckpt_dir}"
        result.num_blocks = model.num_layers
    
    # Create inputs matching production sizes for 480x832 resolution
    inputs = _create_inputs(model, frames)
    result.seq_len = inputs['seq_len']
    
    print(f"\nProfiling: {result.model_info}")
    print(f"Sequence length: {result.seq_len}")
    print(f"Frames: {frames}, Runs: {num_runs}, Warmup: {warmup_runs}")
    print("-" * 60)
    
    # Warmup
    for _ in range(warmup_runs):
        _run_forward_pass(model, inputs)
    
    gc.collect()
    
    # Profile complete forward pass
    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}:")
        run_timings = _profile_forward_pass(model, inputs)
        for name, elapsed in run_timings.items():
            result.add_timing(name, elapsed)
    
    # Calculate total time
    total_times = []
    for run in range(num_runs):
        run_total = sum(
            t.times[run] for t in result.timings.values() 
            if len(t.times) > run and not t.name.startswith("_")
        )
        total_times.append(run_total)
    result.total_time_sec = float(np.mean(total_times))
    
    return result


def _create_synthetic_model(num_blocks: int) -> Tuple[Any, Dict[str, Any]]:
    """Create a small synthetic model for testing."""
    from wan.backends.mlx.model import WanModelMLX
    import mlx.core as mx
    
    config = {
        'model_type': 'i2v',
        'patch_size': (1, 2, 2),
        'text_len': 128,
        'in_dim': 16,
        'dim': 512,
        'ffn_dim': 2048,
        'freq_dim': 64,
        'text_dim': 512,
        'out_dim': 8,
        'num_heads': 8,
        'num_layers': num_blocks,
    }
    
    model = WanModelMLX(**config)
    mx.eval(model.parameters())
    
    return model, config


def _load_full_model(ckpt_dir: Optional[str]) -> Any:
    """Load the full model from checkpoint."""
    from wan.backends.mlx.model import WanModelMLX
    import mlx.core as mx
    
    if ckpt_dir is None:
        ckpt_dir = "models/lingbot-world-base-cam"
    
    model_path = Path(ckpt_dir) / 'low_noise_model'
    print(f"Loading model from {model_path}...")
    
    model = WanModelMLX.from_pretrained(str(model_path), use_cache=True)
    mx.eval(model.parameters())
    
    return model


def _create_inputs(model: Any, frames: int) -> Dict[str, Any]:
    """Create inputs matching production sizes."""
    import mlx.core as mx
    
    # For 480x832 resolution
    lat_h, lat_w = 30, 52  # 480/16, 832/16
    lat_f = (frames - 1) // 4 + 1
    
    # Divide by patch_size product for sequence length
    patch_prod = model.patch_size[0] * model.patch_size[1] * model.patch_size[2]
    seq_len = lat_f * lat_h * lat_w // patch_prod
    
    # Get channel dimensions from patch_embedding weight shape
    # For Conv3d equivalent, weight shape is [out_dim, in_dim * patch_prod] after flattening
    if hasattr(model, 'patch_embedding') and hasattr(model.patch_embedding, 'weight'):
        weight = model.patch_embedding.weight
        # weight shape: [out_dim, in_dim * patch_prod]
        total_in_dim = weight.shape[1] // patch_prod
    else:
        total_in_dim = model.in_dim if hasattr(model, 'in_dim') else 16
    
    # For i2v mode: x is 16 channels (noise latent), y is 20 channels (mask 4 + latent 16)
    # Total is 36 channels for production model
    # For synthetic model with in_dim=16: split evenly
    if total_in_dim == 36:
        x_channels = 16
        y_channels = 20
    else:
        x_channels = total_in_dim // 2
        y_channels = total_in_dim - x_channels
    
    np.random.seed(42)
    x = [mx.array(np.random.randn(x_channels, lat_f, lat_h, lat_w).astype(np.float32))]
    y = [mx.array(np.random.randn(y_channels, lat_f, lat_h, lat_w).astype(np.float32))]
    
    text_dim = model.text_dim if hasattr(model, 'text_dim') else 512
    context = [mx.array(np.random.randn(100, text_dim).astype(np.float32))]
    
    t = mx.array([0.5])
    
    # Camera embeddings
    cam_channels = 6 * 64  # plucker dimension
    c2ws_plucker = [mx.array(np.random.randn(1, cam_channels, lat_f, lat_h, lat_w).astype(np.float32))]
    dit_cond_dict = {"c2ws_plucker_emb": c2ws_plucker}
    
    print(f"  Input channels: x={x_channels}, y={y_channels}, total={total_in_dim}")
    
    return {
        'x': x,
        'y': y,
        't': t,
        'context': context,
        'seq_len': seq_len,
        'dit_cond_dict': dit_cond_dict,
    }


def _run_forward_pass(model: Any, inputs: Dict[str, Any]) -> Any:
    """Run a complete forward pass."""
    import mlx.core as mx
    
    output = model(
        inputs['x'],
        inputs['t'],
        inputs['context'],
        inputs['seq_len'],
        y=inputs['y'],
        dit_cond_dict=inputs['dit_cond_dict'],
    )
    mx.eval(output[0])
    return output


def _profile_forward_pass(model: Any, inputs: Dict[str, Any]) -> Dict[str, float]:
    """Profile each component of the forward pass.
    
    Returns dict of component_name -> elapsed_seconds
    """
    import mlx.core as mx
    import mlx.nn as nn
    
    timings: Dict[str, float] = {}
    
    x_list = inputs['x']
    y_list = inputs['y']
    t = inputs['t']
    context = inputs['context']
    seq_len = inputs['seq_len']
    dit_cond_dict = inputs['dit_cond_dict']
    
    batch_size = len(x_list)
    
    # =========================================================================
    # 1. Input concatenation (i2v mode)
    # =========================================================================
    start = time.perf_counter()
    x_cat = [mx.concatenate([u, v], axis=0) for u, v in zip(x_list, y_list)]
    mx.eval(x_cat[0])
    timings['input_concat'] = time.perf_counter() - start
    
    # =========================================================================
    # 2. Patch embedding
    # =========================================================================
    start = time.perf_counter()
    x_embedded = [model.patch_embedding(mx.expand_dims(u, axis=0)) for u in x_cat]
    mx.eval(x_embedded[0])
    timings['patch_embedding'] = time.perf_counter() - start
    
    # =========================================================================
    # 3. Grid size computation and padding
    # =========================================================================
    start = time.perf_counter()
    grid_sizes_list = []
    for u_orig in x_cat:
        f, h, w = u_orig.shape[1], u_orig.shape[2], u_orig.shape[3]
        f_out = f // model.patch_size[0]
        h_out = h // model.patch_size[1]
        w_out = w // model.patch_size[2]
        grid_sizes_list.append([f_out, h_out, w_out])
    grid_sizes = mx.array(grid_sizes_list, dtype=mx.int32)
    
    x_flat = [u.reshape(u.shape[1], u.shape[2]) for u in x_embedded]
    seq_lens_list = [u.shape[0] for u in x_flat]
    seq_lens = mx.array(seq_lens_list, dtype=mx.int32)
    
    x_padded = []
    for u in x_flat:
        pad_len = seq_len - u.shape[0]
        if pad_len > 0:
            padding = mx.zeros((pad_len, model.dim), dtype=u.dtype)
            u = mx.concatenate([u, padding], axis=0)
        x_padded.append(u)
    x_batched = mx.stack(x_padded, axis=0)
    mx.eval(x_batched, grid_sizes, seq_lens)
    timings['grid_padding'] = time.perf_counter() - start
    
    # =========================================================================
    # 4. Time embedding
    # =========================================================================
    start = time.perf_counter()
    if t.ndim == 1:
        t_broadcast = mx.broadcast_to(mx.expand_dims(t, axis=1), (batch_size, seq_len))
    else:
        t_broadcast = t
    
    # Sinusoidal embedding
    from wan.backends.mlx.model import sinusoidal_embedding_1d_mlx
    t_flat = t_broadcast.flatten()
    t_emb = sinusoidal_embedding_1d_mlx(model.freq_dim, t_flat)
    t_emb = t_emb.reshape(batch_size, seq_len, model.freq_dim)
    
    e = model._time_embedding(t_emb)
    e0 = model._time_projection(e)
    e0 = e0.reshape(batch_size, seq_len, 6, model.dim)
    mx.eval(e0)
    timings['time_embedding'] = time.perf_counter() - start
    
    # =========================================================================
    # 5. Text embedding
    # =========================================================================
    start = time.perf_counter()
    context_padded = []
    for ctx in context:
        pad_len = model.text_len - ctx.shape[0]
        if pad_len > 0:
            padding = mx.zeros((pad_len, ctx.shape[1]), dtype=ctx.dtype)
            ctx = mx.concatenate([ctx, padding], axis=0)
        context_padded.append(ctx)
    context_batched = mx.stack(context_padded, axis=0)
    context_embedded = model._text_embedding(context_batched)
    mx.eval(context_embedded)
    timings['text_embedding'] = time.perf_counter() - start
    
    # =========================================================================
    # 6. Camera embedding
    # =========================================================================
    start = time.perf_counter()
    c2ws_plucker_emb = dit_cond_dict["c2ws_plucker_emb"]
    rearranged = []
    for emb in c2ws_plucker_emb:
        b, c, f, h, w = emb.shape
        pt, ph, pw = model.patch_size
        f_out = f // pt
        h_out = h // ph
        w_out = w // pw
        emb = emb.reshape(b, c, f_out, pt, h_out, ph, w_out, pw)
        emb = mx.transpose(emb, (0, 2, 4, 6, 1, 3, 5, 7))
        emb = emb.reshape(b, f_out * h_out * w_out, -1)
        rearranged.append(emb)
    c2ws_plucker_emb = mx.concatenate(rearranged, axis=1)
    
    c2ws_plucker_emb = model.patch_embedding_wancamctrl(c2ws_plucker_emb)
    c2ws_hidden = nn.silu(model.c2ws_hidden_states_layer1(c2ws_plucker_emb))
    c2ws_hidden = model.c2ws_hidden_states_layer2(c2ws_hidden)
    processed_cam_emb = c2ws_plucker_emb + c2ws_hidden
    mx.eval(processed_cam_emb)
    timings['camera_embedding'] = time.perf_counter() - start
    
    dit_cond_dict_processed = {"c2ws_plucker_emb": processed_cam_emb}
    
    # =========================================================================
    # 7. Profile each attention block
    # =========================================================================
    freqs_cos = model._freqs_cos
    freqs_sin = model._freqs_sin
    
    # Aggregate timings for block sub-components
    block_total_times = []
    self_attn_times = []
    cross_attn_times = []
    ffn_times = []
    modulation_times = []
    camera_injection_times = []
    rope_times = []
    norm_times = []
    
    for i, block in enumerate(model.blocks):
        block_start = time.perf_counter()
        
        # --- Modulation computation ---
        mod_start = time.perf_counter()
        e_mod = mx.expand_dims(block.modulation, axis=0) + e0
        e_parts = [mx.squeeze(e_mod[:, :, j:j+1, :], axis=2) for j in range(6)]
        mx.eval(e_parts[0])
        modulation_times.append(time.perf_counter() - mod_start)
        
        # --- Self-attention ---
        # Pre-norm
        norm_start = time.perf_counter()
        x_normed = block.norm1(x_batched).astype(mx.float32)
        x_mod = x_normed * (1.0 + e_parts[1]) + e_parts[0]
        mx.eval(x_mod)
        norm_times.append(time.perf_counter() - norm_start)
        
        # Self attention components
        attn_start = time.perf_counter()
        b, s, _ = x_mod.shape
        n, d = block.self_attn.num_heads, block.self_attn.head_dim
        
        # Q, K, V projections
        q = block.self_attn.q(x_mod)
        k = block.self_attn.k(x_mod)
        v = block.self_attn.v(x_mod)
        
        if block.self_attn.qk_norm:
            q = block.self_attn.norm_q(q)
            k = block.self_attn.norm_k(k)
        
        q = q.reshape(b, s, n, d)
        k = k.reshape(b, s, n, d)
        v = v.reshape(b, s, n, d)
        mx.eval(q, k, v)
        qkv_time = time.perf_counter() - attn_start
        
        # RoPE
        rope_start = time.perf_counter()
        from wan.backends.mlx.rope import rope_apply_mlx
        q = rope_apply_mlx(q, grid_sizes, freqs_cos, freqs_sin)
        k = rope_apply_mlx(k, grid_sizes, freqs_cos, freqs_sin)
        mx.eval(q, k)
        rope_times.append(time.perf_counter() - rope_start)
        
        # Attention computation
        sdpa_start = time.perf_counter()
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))
        
        attn_output = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=block.self_attn.scale, mask=None
        )
        
        attn_output = mx.transpose(attn_output, (0, 2, 1, 3))
        attn_output = attn_output.reshape(b, s, -1)
        attn_output = block.self_attn.o(attn_output)
        mx.eval(attn_output)
        sdpa_time = time.perf_counter() - sdpa_start
        
        # Residual with modulation
        x_batched = x_batched + attn_output * e_parts[2]
        mx.eval(x_batched)
        
        self_attn_times.append(qkv_time + sdpa_time)
        
        # --- Camera injection ---
        if "c2ws_plucker_emb" in dit_cond_dict_processed:
            cam_start = time.perf_counter()
            cam_emb = dit_cond_dict_processed["c2ws_plucker_emb"]
            c2ws_hidden = nn.silu(block.cam_injector_layer1(cam_emb))
            c2ws_hidden = block.cam_injector_layer2(c2ws_hidden)
            c2ws_hidden = c2ws_hidden + cam_emb
            cam_scale = block.cam_scale_layer(c2ws_hidden)
            cam_shift = block.cam_shift_layer(c2ws_hidden)
            x_batched = (1.0 + cam_scale) * x_batched + cam_shift
            mx.eval(x_batched)
            camera_injection_times.append(time.perf_counter() - cam_start)
        
        # --- Cross-attention ---
        cross_start = time.perf_counter()
        if block._use_norm3:
            x_for_cross = block.norm3(x_batched)
        else:
            x_for_cross = x_batched
        
        cross_out = block.cross_attn(x_for_cross, context_embedded, None)
        x_batched = x_batched + cross_out
        mx.eval(x_batched)
        cross_attn_times.append(time.perf_counter() - cross_start)
        
        # --- FFN ---
        ffn_start = time.perf_counter()
        x_normed = block.norm2(x_batched).astype(mx.float32)
        x_mod = x_normed * (1.0 + e_parts[4]) + e_parts[3]
        y_ffn = block._ffn(x_mod)
        x_batched = x_batched + y_ffn * e_parts[5]
        mx.eval(x_batched)
        ffn_times.append(time.perf_counter() - ffn_start)
        
        block_total_times.append(time.perf_counter() - block_start)
        
        if i < 3 or i >= len(model.blocks) - 2:
            print(f"  Block {i:2d}: {block_total_times[-1]*1000:6.1f}ms "
                  f"(self_attn={self_attn_times[-1]*1000:5.1f}, "
                  f"cross={cross_attn_times[-1]*1000:5.1f}, "
                  f"ffn={ffn_times[-1]*1000:5.1f}, "
                  f"rope={rope_times[-1]*1000:5.1f})")
        elif i == 3:
            print("  ...")
    
    # Store aggregated block timings
    timings['blocks_total'] = sum(block_total_times)
    timings['_self_attn_total'] = sum(self_attn_times)
    timings['_cross_attn_total'] = sum(cross_attn_times)
    timings['_ffn_total'] = sum(ffn_times)
    timings['_rope_total'] = sum(rope_times)
    timings['_modulation_total'] = sum(modulation_times)
    timings['_camera_injection_total'] = sum(camera_injection_times)
    timings['_norm_total'] = sum(norm_times)
    
    # Store per-block average
    timings['blocks_avg'] = float(np.mean(block_total_times))
    
    # =========================================================================
    # 8. Output head
    # =========================================================================
    start = time.perf_counter()
    # Note: head uses e (from time embedding), not e0
    output = model.head(x_batched, e)
    mx.eval(output)
    timings['head'] = time.perf_counter() - start
    
    # =========================================================================
    # 9. Unpatchify
    # =========================================================================
    start = time.perf_counter()
    final_output = model._unpatchify(output, grid_sizes)
    mx.eval(final_output[0])
    timings['unpatchify'] = time.perf_counter() - start
    
    return timings


def print_profile_results(result: ProfileResult) -> None:
    """Print formatted profiling results."""
    print("\n" + "=" * 70)
    print("PROFILING RESULTS")
    print("=" * 70)
    print(f"\nModel: {result.model_info}")
    print(f"Sequence length: {result.seq_len}")
    print(f"Number of blocks: {result.num_blocks}")
    print(f"Number of runs: {result.num_runs}")
    print(f"Total time per forward pass: {result.total_time_sec*1000:.1f}ms")
    
    # Separate main components from sub-components
    main_components = {k: v for k, v in result.timings.items() if not k.startswith('_')}
    sub_components = {k: v for k, v in result.timings.items() if k.startswith('_')}
    
    # Sort by mean time
    sorted_main = sorted(main_components.items(), key=lambda x: -x[1].mean_ms)
    
    total_ms = result.total_time_sec * 1000
    
    print("\n" + "-" * 70)
    print("COMPONENT BREAKDOWN")
    print("-" * 70)
    print(f"{'Component':<25} {'Mean (ms)':<12} {'Std (ms)':<12} {'% of Total':<10}")
    print("-" * 70)
    
    for name, timing in sorted_main:
        if name == 'blocks_avg':
            continue
        pct = (timing.mean_ms / total_ms * 100) if total_ms > 0 else 0
        print(f"{name:<25} {timing.mean_ms:>10.2f}  {timing.std_ms:>10.2f}  {pct:>8.1f}%")
    
    print("\n" + "-" * 70)
    print("BLOCK INTERNALS (aggregated across all blocks)")
    print("-" * 70)
    
    sorted_sub = sorted(sub_components.items(), key=lambda x: -x[1].mean_ms)
    for name, timing in sorted_sub:
        clean_name = name.lstrip('_')
        pct = (timing.mean_ms / total_ms * 100) if total_ms > 0 else 0
        print(f"{clean_name:<25} {timing.mean_ms:>10.2f}  {timing.std_ms:>10.2f}  {pct:>8.1f}%")
    
    # Compute per-block averages
    if result.num_blocks > 0:
        print("\n" + "-" * 70)
        print("PER-BLOCK AVERAGES")
        print("-" * 70)
        
        for name in ['self_attn_total', 'cross_attn_total', 'ffn_total', 
                     'rope_total', 'modulation_total', 'camera_injection_total', 'norm_total']:
            key = f'_{name}'
            if key in sub_components:
                avg_ms = sub_components[key].mean_ms / result.num_blocks
                print(f"{name.replace('_total', ''):<25} {avg_ms:>10.2f} ms/block")
    
    # Identify bottlenecks
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    # Find top 3 bottlenecks
    all_timings = list(main_components.items()) + list(sub_components.items())
    # Filter out aggregates that are already counted
    all_timings = [(k, v) for k, v in all_timings 
                   if k not in ('blocks_total', 'blocks_avg')]
    sorted_all = sorted(all_timings, key=lambda x: -x[1].mean_ms)
    
    print("\nTop bottlenecks:")
    for i, (name, timing) in enumerate(sorted_all[:5]):
        clean_name = name.lstrip('_')
        pct = (timing.mean_ms / total_ms * 100) if total_ms > 0 else 0
        print(f"  {i+1}. {clean_name}: {timing.mean_ms:.1f}ms ({pct:.1f}%)")
    
    # Block vs overhead analysis
    if 'blocks_total' in main_components:
        blocks_pct = main_components['blocks_total'].mean_ms / total_ms * 100
        overhead_ms = total_ms - main_components['blocks_total'].mean_ms
        print(f"\nBlocks account for {blocks_pct:.1f}% of total time")
        print(f"Overhead (embeddings, head, etc.): {overhead_ms:.1f}ms ({100-blocks_pct:.1f}%)")
    
    # FFN vs Attention analysis
    if '_ffn_total' in sub_components and '_self_attn_total' in sub_components:
        ffn_ms = sub_components['_ffn_total'].mean_ms
        self_attn_ms = sub_components['_self_attn_total'].mean_ms
        cross_attn_ms = sub_components.get('_cross_attn_total', ComponentTiming('_')).mean_ms
        total_attn_ms = self_attn_ms + cross_attn_ms
        
        print(f"\nAttention vs FFN:")
        print(f"  Total Attention: {total_attn_ms:.1f}ms ({total_attn_ms/total_ms*100:.1f}%)")
        print(f"  Total FFN: {ffn_ms:.1f}ms ({ffn_ms/total_ms*100:.1f}%)")
        print(f"  Ratio (Attn/FFN): {total_attn_ms/ffn_ms:.2f}x" if ffn_ms > 0 else "")


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Profile MLX Forward Pass',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--mode', choices=['synthetic', 'full'], default='synthetic',
        help='Profile mode: synthetic (small model) or full (real weights)'
    )
    parser.add_argument(
        '--ckpt_dir', default='models/lingbot-world-base-cam',
        help='Model checkpoint directory (for full mode)'
    )
    parser.add_argument(
        '--blocks', type=int, default=4,
        help='Number of blocks for synthetic model (default: 4)'
    )
    parser.add_argument(
        '--frames', type=int, default=11,
        help='Number of video frames (default: 11)'
    )
    parser.add_argument(
        '--runs', type=int, default=3,
        help='Number of profiling runs (default: 3)'
    )
    parser.add_argument(
        '--warmup', type=int, default=1,
        help='Number of warmup runs (default: 1)'
    )
    
    args = parser.parse_args()
    
    try:
        result = profile_mlx_forward(
            mode=args.mode,
            ckpt_dir=args.ckpt_dir if args.mode == 'full' else None,
            num_blocks=args.blocks,
            frames=args.frames,
            num_runs=args.runs,
            warmup_runs=args.warmup,
        )
        print_profile_results(result)
        
    except ImportError as e:
        print(f"Error: {e}")
        print("Make sure MLX is installed: pip install mlx")
        sys.exit(1)
    except Exception as e:
        print(f"Error during profiling: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
