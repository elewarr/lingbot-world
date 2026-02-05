#!/usr/bin/env python3
"""
Benchmark VAE decode with different chunk sizes.

Usage:
    source .venv/bin/activate
    python scripts/benchmark_vae_chunk.py

Results are printed to stdout and appended to docs/BENCHMARKS.md
"""

import os
import sys
import time
import gc

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from wan.modules.vae2_1 import Wan2_1_VAE

# Platform detection
IS_MACOS = sys.platform == 'darwin'
HAS_MPS = IS_MACOS and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
HAS_CUDA = torch.cuda.is_available()


def get_device():
    if HAS_CUDA:
        return torch.device('cuda')
    elif HAS_MPS:
        return torch.device('mps')
    return torch.device('cpu')


def empty_cache():
    if HAS_CUDA:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif HAS_MPS:
        torch.mps.empty_cache()
        torch.mps.synchronize()


def get_memory_mb():
    """Get current memory usage in MB."""
    if HAS_CUDA:
        return torch.cuda.memory_allocated() / 1024 / 1024
    elif HAS_MPS:
        # MPS doesn't have direct memory query, return 0
        return 0
    return 0


def benchmark_vae_decode(vae, latent_frames, chunk_sizes, num_runs=3):
    """
    Benchmark VAE decode with different chunk sizes.
    
    Args:
        vae: Loaded VAE model
        latent_frames: Number of latent frames to decode
        chunk_sizes: List of chunk sizes to test (0 = no chunking)
        num_runs: Number of runs per configuration
    
    Returns:
        dict: Results with timing for each chunk size
    """
    device = vae.device
    dtype = vae.dtype
    
    # Create dummy latent tensor
    # Shape: [C, T, H, W] where C=16 (z_dim), H=60, W=104 for 480x832
    latent = torch.randn(16, latent_frames, 60, 104, dtype=dtype, device=device)
    
    results = {}
    
    for chunk_size in chunk_sizes:
        times = []
        peak_memory = 0
        
        for run in range(num_runs):
            empty_cache()
            gc.collect()
            
            start_mem = get_memory_mb()
            start_time = time.perf_counter()
            
            if chunk_size == 0 or latent_frames <= chunk_size:
                # No chunking
                _ = vae.decode([latent])
            else:
                # Chunked decode
                video_chunks = []
                for i in range(0, latent_frames, chunk_size):
                    end_idx = min(i + chunk_size, latent_frames)
                    chunk = latent[:, i:end_idx, :, :]
                    chunk_video = vae.decode([chunk])[0]
                    video_chunks.append(chunk_video)
                    if HAS_MPS:
                        empty_cache()
                
                _ = torch.cat(video_chunks, dim=1)
                del video_chunks
            
            end_time = time.perf_counter()
            end_mem = get_memory_mb()
            
            times.append(end_time - start_time)
            peak_memory = max(peak_memory, end_mem - start_mem)
            
            empty_cache()
            gc.collect()
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        
        label = f"chunk_{chunk_size}" if chunk_size > 0 else "no_chunk"
        results[label] = {
            'chunk_size': chunk_size,
            'avg_time': avg_time,
            'min_time': min_time,
            'peak_memory_mb': peak_memory,
            'times': times,
        }
        
        print(f"  {label}: avg={avg_time:.2f}s, min={min_time:.2f}s, peak_mem={peak_memory:.0f}MB")
    
    return results


def main():
    device = get_device()
    print(f"Device: {device}")
    print(f"Platform: {'MPS' if HAS_MPS else 'CUDA' if HAS_CUDA else 'CPU'}")
    
    # Load VAE
    vae_path = 'models/lingbot-world-base-cam/Wan2.1_VAE.pth'
    if not os.path.exists(vae_path):
        print(f"ERROR: VAE not found at {vae_path}")
        print("Please ensure model weights are downloaded.")
        sys.exit(1)
    
    print(f"\nLoading VAE from {vae_path}...")
    dtype = torch.float32  # MPS requires float32
    vae = Wan2_1_VAE(vae_pth=vae_path, dtype=dtype, device=device)
    print("VAE loaded.")
    
    # Test configurations
    # latent_frames -> video_frames: multiply by 4
    # 11 latent = 41 video frames (production)
    # 21 latent = 81 video frames (long)
    
    test_configs = [
        (11, [0, 2, 4, 6, 8, 11]),  # 41 video frames
    ]
    
    print("\n" + "="*60)
    print("VAE Decode Chunk Size Benchmark")
    print("="*60)
    
    all_results = {}
    
    for latent_frames, chunk_sizes in test_configs:
        video_frames = latent_frames * 4
        print(f"\n{latent_frames} latent frames ({video_frames} video frames):")
        print("-" * 40)
        
        results = benchmark_vae_decode(vae, latent_frames, chunk_sizes, num_runs=3)
        all_results[latent_frames] = results
    
    # Print summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    for latent_frames, results in all_results.items():
        video_frames = latent_frames * 4
        print(f"\n{video_frames} video frames:")
        
        # Find baseline (no chunking or largest chunk)
        baseline_key = 'no_chunk' if 'no_chunk' in results else max(results.keys(), key=lambda k: results[k]['chunk_size'])
        baseline_time = results[baseline_key]['avg_time']
        
        for label, data in sorted(results.items(), key=lambda x: x[1]['chunk_size']):
            speedup = baseline_time / data['avg_time'] if data['avg_time'] > 0 else 0
            chunk_str = f"chunk={data['chunk_size']}" if data['chunk_size'] > 0 else "no chunk"
            print(f"  {chunk_str:12s}: {data['avg_time']:.2f}s (speedup: {speedup:.2f}x)")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
