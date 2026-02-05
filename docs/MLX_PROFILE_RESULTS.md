# MLX Forward Pass Profiling Results

> Profiled on Apple Silicon with MLX backend

## Summary

The profiling script identifies where time is spent in the MLX forward pass to guide optimization efforts.

## Key Findings

### Bottleneck Ranking (Full 14B Model, 480x832 resolution)

| Component | % of Time | Optimization Priority |
|-----------|-----------|----------------------|
| FFN (feed-forward) | 27-30% | HIGH - consider quantization |
| Self-Attention | 25-27% | HIGH - already using SDPA |
| Camera Injection | 19-21% | MEDIUM - 4 linear layers per block |
| Cross-Attention | 15-20% | MEDIUM - already using SDPA |
| RoPE | 1.7-1.9% | LOW - already optimized |
| Modulation | 0.9-2.6% | LOW |
| Norms | 0.6-0.7% | LOW - using mx.fast kernels |

### Scaling with Sequence Length

| Frames | Seq Len | Forward Pass | Self-Attn/block | FFN/block |
|--------|---------|--------------|-----------------|-----------|
| 5 | 780 | 2.4s | 15.4ms | 16.2ms |
| 11 | 1170 | 3.0s | 19.5ms | 19.0ms |
| 21 | 2340 | 5.1s | 34.2ms | 38.1ms |

Self-attention scales ~O(n^2) with sequence length, while FFN scales ~O(n).

### Time Distribution (21 frames)

```
Blocks: 96.3%
  ├── FFN: 29.7%
  ├── Self-Attention: 26.8%
  ├── Camera Injection: 20.5%
  ├── Cross-Attention: 15.4%
  ├── RoPE: 1.9%
  ├── Modulation: 0.9%
  └── Norms: 0.7%
Overhead: 3.7%
  ├── Time Embedding: 0.8%
  ├── Camera Embedding: 0.3%
  ├── Text Embedding: 0.1%
  └── Other: <0.1%
```

## Optimization Recommendations

### 1. Quantization (HIGH PRIORITY)
- FFN layers (27-30% of time) are memory-bandwidth bound
- INT8/INT4 quantization can significantly speed up matmuls
- Target: `ffn_linear1`, `ffn_linear2` in each block

### 2. Camera Injection (MEDIUM PRIORITY)
- 4 linear layers per block: `cam_injector_layer1/2`, `cam_scale/shift_layer`
- Taking 19-21% of time
- Could potentially be fused or quantized

### 3. Memory Bandwidth Optimization
- The model is likely memory-bandwidth bound on Apple Silicon
- Quantization reduces memory traffic and improves throughput
- Fused kernels reduce memory round-trips

### 4. Attention Optimization
- Already using `mx.fast.scaled_dot_product_attention`
- Limited room for improvement without custom Metal kernels
- Consider chunked attention for very long sequences

## Running the Profiler

```bash
# Quick test with synthetic model
python scripts/profile_mlx.py --mode synthetic --blocks 8

# Full model profiling (requires model weights)
python scripts/profile_mlx.py --mode full --frames 11 --runs 3

# Production-like test
python scripts/profile_mlx.py --mode full --frames 41 --runs 2
```

## Notes

- All timings measured with `mx.eval()` for synchronization
- Warmup runs performed before timing
- Profiling adds overhead; actual inference may be slightly faster
- Results are device-specific (tested on Apple Silicon)
