#!/usr/bin/env python3
"""
Benchmark quantization on LingBot-World DiT model.

Usage:
    source .venv/bin/activate
    python scripts/benchmark_quantization.py

This script tests different quantization types (NF4, INT8, FP8) and measures:
- Memory usage
- Quantization time
"""

import argparse
import gc
import logging
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

# Platform detection
IS_MACOS = sys.platform == 'darwin'
HAS_MPS = IS_MACOS and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()


def empty_cache():
    if HAS_MPS:
        torch.mps.empty_cache()
        torch.mps.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def get_memory_mb(model):
    """Get model memory in MB, including quantized weights."""
    total = 0
    for name, param in model.named_parameters():
        total += param.numel() * param.element_size()
    for name, buffer in model.named_buffers():
        total += buffer.numel() * buffer.element_size()
    return total / 1024 / 1024


def get_full_model_size_mb(model):
    """Get model size including all tensors (for quantized models)."""
    import sys
    total = 0
    seen = set()
    
    def add_tensor(t):
        nonlocal total
        if id(t) not in seen:
            seen.add(id(t))
            total += t.numel() * t.element_size()
    
    for param in model.parameters():
        add_tensor(param.data)
    for buffer in model.buffers():
        add_tensor(buffer)
    
    # For quantized layers, also count internal buffers
    for module in model.modules():
        if hasattr(module, 'weight_packed'):
            add_tensor(module.weight_packed)
        if hasattr(module, 'weight_absmax'):
            add_tensor(module.weight_absmax)
        if hasattr(module, 'weight_quant'):
            add_tensor(module.weight_quant)
        if hasattr(module, 'weight_scales'):
            add_tensor(module.weight_scales)
    
    return total / 1024 / 1024


def main():
    parser = argparse.ArgumentParser(description='Benchmark quantization')
    parser.add_argument('--ckpt_dir', default='models/lingbot-world-base-cam',
                        help='Model checkpoint directory')
    parser.add_argument('--quant_types', nargs='+', default=['none', 'nf4', 'int8'],
                        help='Quantization types to test')
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    device = 'mps' if HAS_MPS else 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Device: {device}")
    logging.info(f"Platform: {'MPS' if HAS_MPS else 'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Check for mps-bitsandbytes
    try:
        from wan.utils.quantize import quantize_dit_model, HAS_MPS_BNB
        if not HAS_MPS_BNB:
            logging.error("mps-bitsandbytes not available. Install with: pip install mps-bitsandbytes")
            logging.info("Running baseline only...")
            args.quant_types = ['none']
    except ImportError as e:
        logging.error(f"Import error: {e}")
        return
    
    # Check model exists
    low_noise_dir = os.path.join(args.ckpt_dir, 'low_noise_model')
    if not os.path.exists(low_noise_dir):
        logging.error(f"Model not found at {low_noise_dir}")
        logging.info("Please ensure model weights are downloaded.")
        return
    
    results = {}
    
    for quant_type in args.quant_types:
        logging.info(f"\n{'='*60}")
        logging.info(f"Testing: {quant_type.upper()}")
        logging.info('='*60)
        
        empty_cache()
        
        # Load model using diffusers' from_pretrained
        logging.info("Loading model...")
        load_start = time.perf_counter()
        
        from wan.modules.model import WanModel
        
        # Load low_noise_model (one of the two DiT models)
        model = WanModel.from_pretrained(
            args.ckpt_dir,
            subfolder='low_noise_model',
            torch_dtype=torch.float32,  # MPS needs float32
        )
        
        load_time = time.perf_counter() - load_start
        logging.info(f"Model loaded in {load_time:.2f}s")
        
        # Get baseline memory
        mem_before = get_memory_mb(model)
        logging.info(f"Memory (CPU): {mem_before:.1f} MB")
        
        # Move to device and optionally quantize
        if quant_type == 'none':
            model = model.to(device)
            mem_after = mem_before  # For baseline, same as CPU
            full_size = mem_before
        else:
            # Quantize
            quant_start = time.perf_counter()
            model = quantize_dit_model(model, quant_type=quant_type, device=device)
            quant_time = time.perf_counter() - quant_start
            logging.info(f"Quantization time: {quant_time:.2f}s")
            mem_after = get_memory_mb(model)
        
        savings = (1 - mem_after / mem_before) * 100 if mem_before > 0 else 0
        
        # Also get full size including quantized buffers
        full_size = get_full_model_size_mb(model)
        logging.info(f"Memory (device): {mem_after:.1f} MB (params), {full_size:.1f} MB (full)")
        logging.info(f"Savings: {savings:.1f}% (params)")
        
        results[quant_type] = {
            'load_time': load_time,
            'memory_mb': full_size,
            'param_memory_mb': mem_after,
            'savings_pct': (1 - full_size / mem_before) * 100 if mem_before > 0 else 0,
        }
        
        # Cleanup
        del model
        empty_cache()
    
    # Summary
    logging.info(f"\n{'='*60}")
    logging.info("SUMMARY")
    logging.info('='*60)
    logging.info(f"{'Type':<10} {'Memory (MB)':<15} {'Savings':<10}")
    logging.info('-'*40)
    
    for quant_type, data in results.items():
        mem = data['memory_mb']
        savings = data['savings_pct']
        logging.info(f"{quant_type:<10} {mem:<15.1f} {savings:.1f}%")
    
    logging.info("\nDone!")


if __name__ == '__main__':
    main()
