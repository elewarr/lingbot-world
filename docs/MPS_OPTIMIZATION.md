# MPS (Apple Silicon) Optimization Guide

## Current Performance Baseline

| Config | Time/Step | VAE Decode | Total | Notes |
|--------|-----------|------------|-------|-------|
| 5 frames, 480×832, fp32 | ~18s | ~2s | ~1.5 min | Fast, good for testing |
| 41 frames, 480×832, fp32 | ~160s | **~19s** | ~13 min | Production (with chunk=2) |
| 81 frames, 480×832, fp32 | ~300s (est) | ~38s (est) | ~25 min | Full length |

**Hardware:** Mac Studio M3 Ultra, 512GB unified memory
**VAE Improvement:** chunk_size=2 provides **1.38x speedup** over no chunking

---

## Optimization Options (Least to Most Effort)

### 1. ✅ Already Implemented
- **Offloading**: Enabled by default, moves unused model to CPU
- **SDPA Fallback**: Uses PyTorch's scaled_dot_product_attention instead of flash_attn
- **Float32 dtype**: Required because MPS doesn't support bfloat16, and float16 has accumulator issues

### 2. 🔧 Low Effort - Environment Tuning

#### 2a. VAE Decode Chunk Size
The VAE decoder processes video frames in chunks to avoid OOM. Tune via environment variables:

```bash
# Default: 2 for MPS (benchmarked optimal), 16 for CUDA
export VAE_CHUNK_SIZE=2       # Smaller = faster on MPS (counterintuitively)
export VAE_CHUNK_THRESHOLD=2  # When to start chunking (default: same as chunk_size)
```

**Benchmark Results (M3 Ultra, 44 video frames / 11 latent frames):**

| Chunk Size | Time | Speedup vs No Chunk |
|------------|------|---------------------|
| No chunk | 25.71s | 1.00x (baseline) |
| 2 | **18.64s** | **1.38x** |
| 4 | 23.60s | 1.09x |
| 6 | 25.38s | 1.01x |
| 8 | 25.36s | 1.01x |

**Why smaller chunks are faster on MPS:**
- Reduced memory pressure allows better Metal command buffer pipelining
- More frequent cache clearing prevents memory fragmentation
- MPS memory allocator performs better with smaller allocations

**Trade-off:** Chunk size 2 is optimal for speed. Larger chunks may be needed if you see numerical issues.

#### 2b. MPS Memory Allocation Strategy
```bash
# Add to generate.py or set before running
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0  # Disable memory limit
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.0   # Aggressive memory use
```

**Potential gain:** 5-10% (reduces memory allocation overhead)

#### 2b. Disable Gradient Checkpointing Overhead
Already using `model.eval().requires_grad_(False)` - no gradients computed.

#### 2c. Reduce Sample Steps
```bash
--sample_steps 20  # Instead of 70, quality tradeoff
```

**Potential gain:** Linear (70→20 = 3.5x faster)

### 3. 🔧 Medium Effort - Code Changes

#### 3a. Mixed Precision for Intermediate Tensors
Keep model weights in float32 but compute attention in float16 where safe.

**File:** `wan/modules/attention.py`
```python
def _sdpa_attention(...):
    # Cast to float16 for attention computation, back to float32 for output
    q_t = q.transpose(1, 2).to(torch.float16)  # Attention in fp16
    k_t = k.transpose(1, 2).to(torch.float16)
    v_t = v.transpose(1, 2).to(torch.float16)
    
    out = torch.nn.functional.scaled_dot_product_attention(...)
    
    return out.transpose(1, 2).to(out_dtype)  # Back to fp32
```

**Risk:** May cause accumulator dtype mismatch on some MPS operations
**Potential gain:** 20-40% memory, 10-20% speed

#### 3b. Chunked Attention for Long Sequences
Process attention in chunks to reduce peak memory:

```python
def chunked_sdpa(q, k, v, chunk_size=1024):
    """Process attention in chunks for memory efficiency."""
    B, L, N, D = q.shape
    outputs = []
    for i in range(0, L, chunk_size):
        q_chunk = q[:, i:i+chunk_size]
        # Full K, V for each Q chunk (causal would need masking)
        out_chunk = F.scaled_dot_product_attention(q_chunk, k, v)
        outputs.append(out_chunk)
    return torch.cat(outputs, dim=1)
```

**Potential gain:** Enables larger batch/frame sizes, marginal speed impact

#### 3c. Disable Offloading (512GB systems only)
For systems with enough memory, keeping both models on MPS avoids CPU↔GPU transfers:

```bash
--offload_model False
```

**Note:** Benchmarks showed this was SLOWER on M3 Ultra, possibly due to memory bandwidth. Test on your system.

### 4. 🔨 High Effort - Architectural Changes

#### 4a. ✅ Quantization with mps-bitsandbytes (IMPLEMENTED)

We now support 4-bit and 8-bit quantization using `mps-bitsandbytes`:

```bash
pip install mps-bitsandbytes
```

```python
from wan.utils.quantize import quantize_dit_model

# Load model normally
model = WanModel.from_pretrained(...)

# Quantize to NF4 (4-bit) - best memory savings
model = quantize_dit_model(model, quant_type='nf4', device='mps')

# Or INT8 (8-bit) - better precision
model = quantize_dit_model(model, quant_type='int8', device='mps')
```

**Benchmark Results (M3 Ultra, low_noise_model - 14B params):**

| Quant Type | Memory | Savings | Quant Time | Notes |
|------------|--------|---------|------------|-------|
| None (fp32) | 70.7 GB | - | - | Baseline |
| **NF4 (4-bit)** | **8.8 GB** | **87.5%** | 38s | Best for memory |
| INT8 (8-bit) | 17.7 GB | 75.0% | 7s | Better precision |

**Note:** Total model has TWO DiT models (low_noise + high_noise), so full quantized inference would use:
- NF4: ~18 GB (vs 141 GB unquantized)
- INT8: ~35 GB (vs 141 GB unquantized)

To run the benchmark yourself:
```bash
python scripts/benchmark_quantization.py --quant_types none nf4 int8
```

#### 4b. MLX Backend (Major Rewrite)
Apple's MLX framework is optimized for Apple Silicon:

```python
# MLX equivalent would look like:
import mlx.core as mx
import mlx.nn as nn

class WanModelMLX(nn.Module):
    # Rewrite model using MLX primitives
    pass
```

**Requirements:**
- Full model rewrite (~2000 lines)
- New attention implementation
- Weight conversion from PyTorch

**Potential gain:** 2-5x speed, native Metal optimization

#### 4c. Core ML Conversion
Convert to Core ML for Apple's neural engine:

```python
import coremltools as ct

# Convert PyTorch model to Core ML
mlmodel = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=(1, 16, F, H, W))],
    compute_units=ct.ComputeUnit.ALL  # Use GPU + Neural Engine
)
```

**Requirements:**
- Model tracing (complex with dynamic shapes)
- Custom ops may not be supported
- Significant engineering effort

**Potential gain:** 2-4x speed, leverages Neural Engine

---

## Recommended Implementation Order

1. **Quick wins** (do now):
   - Set MPS memory environment variables
   - Use `--sample_steps 20-30` for faster iteration

2. **Short-term** (1-2 days):
   - Implement mixed precision for attention (3a)
   - Add chunked attention option (3b)

3. **Medium-term** (1 week):
   - INT8 quantization with calibration (4a)
   - Profile and optimize hotspots

4. **Long-term** (2-4 weeks):
   - MLX port for native performance (4b)
   - Or Core ML for Neural Engine (4c)

---

## Profiling Commands

```bash
# Profile MPS operations
python -c "
import torch
torch.mps.profiler.start()
# ... run inference ...
torch.mps.profiler.stop()
"

# Memory tracking
watch -n 1 'sudo powermetrics --samplers gpu_power -i 1000 -n 1 2>/dev/null | grep -E "GPU|Power"'
```

---

## Known Limitations

1. **No bfloat16**: MPS doesn't support bfloat16, must use float32
2. **No flash_attn**: flash_attn CUDA kernels don't run on MPS, using SDPA fallback
3. **Accumulator dtype**: MPS matmul requires matching accumulator/destination dtypes
4. **No FSDP**: Distributed training not applicable to single-device MPS

---

## Benchmark Commands

```bash
# Quick test (5 frames, 5 steps)
python generate.py --task i2v-A14B --size "480*832" \
    --ckpt_dir models/lingbot-world-base-cam \
    --image examples/00/image.jpg --action_path examples/00 \
    --frame_num 5 --sample_steps 5 --prompt "test" --offload_model True

# Production (41 frames, 20 steps)
python generate.py --task i2v-A14B --size "480*832" \
    --ckpt_dir models/lingbot-world-base-cam \
    --image examples/00/image.jpg --action_path examples/00 \
    --frame_num 41 --sample_steps 20 --prompt "your prompt" --offload_model True
```
