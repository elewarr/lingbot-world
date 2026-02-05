#!/usr/bin/env python3
"""MLX Backend Benchmarks - Compare MLX vs PyTorch/MPS inference.

Measures and compares:
- Time per denoising step
- Total generation time
- Memory usage
- Different quantization levels (None, INT8, NF4)
- Different frame counts (5, 41, 81)

Usage:
    source .venv/bin/activate
    
    # Quick synthetic benchmark (no model required)
    python scripts/benchmark_mlx.py --mode synthetic
    
    # Full benchmark with model
    python scripts/benchmark_mlx.py --ckpt_dir models/lingbot-world-base-cam
    
    # Specific configurations
    python scripts/benchmark_mlx.py --frames 41 --steps 20 --quant int8

Output written to docs/MLX_BENCHMARKS.md
"""

import argparse
import gc
import json
import logging
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


# =============================================================================
# Type Definitions
# =============================================================================

QuantType = Literal['none', 'int8', 'int4', 'nf4']


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""
    frames: int
    steps: int
    quant: Optional[str]
    backend: Literal['pytorch', 'mlx']
    warmup_runs: int = 1
    
    def __post_init__(self) -> None:
        if self.quant == 'none':
            self.quant = None


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    config: BenchmarkConfig
    time_per_step_sec: float
    total_time_sec: float
    memory_gb: float
    model_memory_gb: float
    peak_memory_gb: float
    error: Optional[str] = None
    
    @property
    def speedup(self) -> Optional[float]:
        """Speedup vs baseline (calculated after all results are collected)."""
        return None  # Set later via comparison


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""
    hardware: str
    model_info: str
    resolution: str
    timestamp: str
    baseline_total_sec: Optional[float] = None
    results: List[BenchmarkResult] = field(default_factory=list)


# =============================================================================
# Hardware Detection
# =============================================================================

def get_hardware_info() -> str:
    """Get hardware description string."""
    try:
        if sys.platform == 'darwin':
            # Get Mac model and chip info
            model_cmd = subprocess.run(
                ['sysctl', '-n', 'hw.model'],
                capture_output=True, text=True, check=True
            )
            model = model_cmd.stdout.strip()
            
            # Get chip name
            chip_cmd = subprocess.run(
                ['sysctl', '-n', 'machdep.cpu.brand_string'],
                capture_output=True, text=True, check=True
            )
            chip = chip_cmd.stdout.strip()
            
            # Get memory
            mem_cmd = subprocess.run(
                ['sysctl', '-n', 'hw.memsize'],
                capture_output=True, text=True, check=True
            )
            mem_bytes = int(mem_cmd.stdout.strip())
            mem_gb = mem_bytes / (1024**3)
            
            return f"Apple Silicon ({chip}), {mem_gb:.0f}GB"
        else:
            return f"{platform.processor()}, {platform.machine()}"
    except Exception as e:
        return f"Unknown ({e})"


def get_process_memory_gb() -> float:
    """Get current process memory usage in GB."""
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_maxrss / (1024**3) if sys.platform == 'darwin' else usage.ru_maxrss / (1024**2)
    except Exception:
        return 0.0


def get_system_memory_info() -> Tuple[float, float]:
    """Get system memory (used, total) in GB."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return mem.used / (1024**3), mem.total / (1024**3)
    except ImportError:
        return 0.0, 0.0


# =============================================================================
# Synthetic Benchmarks (No Model Required)
# =============================================================================

def run_synthetic_mlx_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    """Run synthetic MLX benchmark using small model components."""
    try:
        import mlx.core as mx
        import mlx.nn as nn
        from wan.backends.mlx.model import WanModelMLX
        from wan.backends.mlx.quantize import quantize_model, get_model_memory_bytes
    except ImportError as e:
        return BenchmarkResult(
            config=config,
            time_per_step_sec=0.0,
            total_time_sec=0.0,
            memory_gb=0.0,
            model_memory_gb=0.0,
            peak_memory_gb=0.0,
            error=f"MLX not available: {e}"
        )
    
    logging.info(f"Synthetic MLX: frames={config.frames}, steps={config.steps}, quant={config.quant}")
    
    # Create a small model for testing
    model_config: Dict[str, Any] = {
        'model_type': 'i2v',
        'patch_size': (1, 2, 2),
        'text_len': 128,
        'in_dim': 16,  # 8 from x + 8 from y
        'dim': 512,    # Smaller than production 2048
        'ffn_dim': 2048,
        'freq_dim': 64,
        'text_dim': 512,
        'out_dim': 8,
        'num_heads': 8,
        'num_layers': 4,  # Smaller than production 32
    }
    
    model = WanModelMLX(**model_config)  # type: ignore[arg-type]
    mx.eval(model.parameters())
    
    # Get model memory before quantization
    model_mem_before = get_model_memory_bytes(model) / (1024**3)
    
    # Apply quantization if requested
    if config.quant:
        quant_type = 'int4' if config.quant == 'nf4' else config.quant
        model = quantize_model(model, quant_type=quant_type)  # type: ignore[arg-type]
        mx.eval(model.parameters())
    
    model_mem_after = get_model_memory_bytes(model) / (1024**3)
    
    # Create synthetic inputs
    batch_size = 1
    in_dim: int = model_config['in_dim']
    x_channels = in_dim // 2
    pt: int = model_config['patch_size'][0]
    ph: int = model_config['patch_size'][1]
    pw: int = model_config['patch_size'][2]
    text_dim: int = model_config['text_dim']
    
    # Scale spatial dims based on frame count (approximate production sizing)
    if config.frames <= 5:
        f, h, w = 4, 16, 32
    elif config.frames <= 41:
        f, h, w = 12, 30, 52
    else:
        f, h, w = 24, 30, 52
    
    seq_len = (f // pt) * (h // ph) * (w // pw)
    
    np.random.seed(42)
    x = [mx.array(np.random.randn(x_channels, f, h, w).astype(np.float32))]
    y = [mx.array(np.random.randn(x_channels, f, h, w).astype(np.float32))]
    context = [mx.array(np.random.randn(100, text_dim).astype(np.float32))]
    
    step_times: List[float] = []
    
    # Warmup
    for _ in range(config.warmup_runs):
        t = mx.array([0.5])
        _ = model(x, t, context, seq_len, y=y)
        mx.eval(_[0])
    
    # Benchmark denoising steps
    total_start = time.perf_counter()
    for step in range(config.steps):
        timestep = (config.steps - step) / config.steps
        t = mx.array([timestep])
        
        step_start = time.perf_counter()
        output = model(x, t, context, seq_len, y=y)
        mx.eval(output[0])
        step_times.append(time.perf_counter() - step_start)
        
        # Simulate scheduler step (minimal overhead)
        x = [output[0][:x_channels]]
    
    total_time = time.perf_counter() - total_start
    avg_step_time = float(np.mean(step_times))
    
    process_mem = get_process_memory_gb()
    
    return BenchmarkResult(
        config=config,
        time_per_step_sec=avg_step_time,
        total_time_sec=total_time,
        memory_gb=process_mem,
        model_memory_gb=model_mem_after,
        peak_memory_gb=process_mem,
    )


def run_synthetic_pytorch_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    """Run synthetic PyTorch/MPS benchmark using small model components."""
    try:
        import torch
        from wan.modules.model import WanModel
    except ImportError as e:
        return BenchmarkResult(
            config=config,
            time_per_step_sec=0.0,
            total_time_sec=0.0,
            memory_gb=0.0,
            model_memory_gb=0.0,
            peak_memory_gb=0.0,
            error=f"PyTorch not available: {e}"
        )
    
    logging.info(f"Synthetic PyTorch: frames={config.frames}, steps={config.steps}")
    
    # Determine device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        device_type = 'mps'
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        device_type = 'cuda'
    else:
        device = torch.device('cpu')
        device_type = 'cpu'
    
    logging.info(f"  Using device: {device_type}")
    
    # Create a small model for testing
    model_config: Dict[str, Any] = {
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
        'num_layers': 4,
    }
    
    model = WanModel(**model_config)  # type: ignore[arg-type]
    model.eval()
    model.requires_grad_(False)
    model.to(device=device, dtype=torch.float32)
    
    # Get model memory
    model_mem = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**3)
    
    # Create synthetic inputs
    batch_size = 1
    in_dim: int = model_config['in_dim']
    x_channels = in_dim // 2
    pt: int = model_config['patch_size'][0]
    ph: int = model_config['patch_size'][1]
    pw: int = model_config['patch_size'][2]
    text_dim: int = model_config['text_dim']
    
    if config.frames <= 5:
        f, h, w = 4, 16, 32
    elif config.frames <= 41:
        f, h, w = 12, 30, 52
    else:
        f, h, w = 24, 30, 52
    
    seq_len = (f // pt) * (h // ph) * (w // pw)
    
    torch.manual_seed(42)
    x = [torch.randn(x_channels, f, h, w, device=device, dtype=torch.float32)]
    y = [torch.randn(x_channels, f, h, w, device=device, dtype=torch.float32)]
    context = [torch.randn(100, text_dim, device=device, dtype=torch.float32)]
    
    step_times: List[float] = []
    
    # Warmup
    with torch.no_grad():
        for _ in range(config.warmup_runs):
            t = torch.tensor([0.5], device=device)
            _ = model(x, t=t, context=context, seq_len=seq_len, y=y)
            if device_type == 'mps':
                torch.mps.synchronize()
            elif device_type == 'cuda':
                torch.cuda.synchronize()
    
    # Benchmark denoising steps
    total_start = time.perf_counter()
    with torch.no_grad():
        for step in range(config.steps):
            timestep = (config.steps - step) / config.steps
            t = torch.tensor([timestep], device=device)
            
            step_start = time.perf_counter()
            output = model(x, t=t, context=context, seq_len=seq_len, y=y)
            if device_type == 'mps':
                torch.mps.synchronize()
            elif device_type == 'cuda':
                torch.cuda.synchronize()
            step_times.append(time.perf_counter() - step_start)
            
            x = [output[0][:x_channels]]
    
    total_time = time.perf_counter() - total_start
    avg_step_time = float(np.mean(step_times))
    
    process_mem = get_process_memory_gb()
    
    return BenchmarkResult(
        config=config,
        time_per_step_sec=avg_step_time,
        total_time_sec=total_time,
        memory_gb=process_mem,
        model_memory_gb=model_mem,
        peak_memory_gb=process_mem,
    )


# =============================================================================
# Full Model Benchmarks
# =============================================================================

def run_full_mlx_benchmark(
    config: BenchmarkConfig,
    ckpt_dir: str,
) -> BenchmarkResult:
    """Run full MLX benchmark with actual model weights."""
    try:
        import mlx.core as mx
        from wan.backends.mlx.model import WanModelMLX
        from wan.backends.mlx.quantize import quantize_model, get_model_memory_bytes
    except ImportError as e:
        return BenchmarkResult(
            config=config,
            time_per_step_sec=0.0,
            total_time_sec=0.0,
            memory_gb=0.0,
            model_memory_gb=0.0,
            peak_memory_gb=0.0,
            error=f"MLX not available: {e}"
        )
    
    logging.info(f"Full MLX: frames={config.frames}, steps={config.steps}, quant={config.quant}")
    
    # Load model from checkpoint
    model_path = Path(ckpt_dir) / 'low_noise_model'
    if not model_path.exists():
        return BenchmarkResult(
            config=config,
            time_per_step_sec=0.0,
            total_time_sec=0.0,
            memory_gb=0.0,
            model_memory_gb=0.0,
            peak_memory_gb=0.0,
            error=f"Model not found at {model_path}"
        )
    
    load_start = time.perf_counter()
    model = WanModelMLX.from_pretrained(str(model_path), use_cache=True)
    mx.eval(model.parameters())
    load_time = time.perf_counter() - load_start
    logging.info(f"  Model loaded in {load_time:.1f}s")
    
    model_mem_before = get_model_memory_bytes(model) / (1024**3)
    
    # Apply quantization if requested
    if config.quant:
        quant_type = 'int4' if config.quant == 'nf4' else config.quant
        quant_start = time.perf_counter()
        model = quantize_model(model, quant_type=quant_type)  # type: ignore[arg-type]
        mx.eval(model.parameters())
        quant_time = time.perf_counter() - quant_start
        logging.info(f"  Quantized to {config.quant} in {quant_time:.1f}s")
    
    model_mem_after = get_model_memory_bytes(model) / (1024**3)
    logging.info(f"  Model memory: {model_mem_before:.1f}GB -> {model_mem_after:.1f}GB")
    
    # Create synthetic inputs matching production sizes
    # For 480x832 resolution
    lat_h, lat_w = 30, 52  # 480/16, 832/16
    lat_f = (config.frames - 1) // 4 + 1
    seq_len = lat_f * lat_h * lat_w // 4  # divide by patch_size product
    
    in_dim = 16  # VAE latent channels
    x_channels = in_dim // 2  # For concatenated i2v mode
    
    np.random.seed(42)
    x = [mx.array(np.random.randn(in_dim, lat_f, lat_h, lat_w).astype(np.float32))]
    y = [mx.array(np.random.randn(in_dim, lat_f, lat_h, lat_w).astype(np.float32))]
    context = [mx.array(np.random.randn(256, 4096).astype(np.float32))]  # T5 output
    
    step_times: List[float] = []
    
    # Warmup
    for _ in range(config.warmup_runs):
        t = mx.array([0.5])
        output = model(x, t, context, seq_len, y=y)
        mx.eval(output[0])
    
    # Benchmark
    gc.collect()
    total_start = time.perf_counter()
    
    for step in range(config.steps):
        timestep = (config.steps - step) / config.steps
        t = mx.array([timestep])
        
        step_start = time.perf_counter()
        output = model(x, t, context, seq_len, y=y)
        mx.eval(output[0])
        step_times.append(time.perf_counter() - step_start)
    
    total_time = time.perf_counter() - total_start
    avg_step_time = float(np.mean(step_times))
    
    process_mem = get_process_memory_gb()
    
    # Cleanup
    del model
    gc.collect()
    
    return BenchmarkResult(
        config=config,
        time_per_step_sec=avg_step_time,
        total_time_sec=total_time,
        memory_gb=process_mem,
        model_memory_gb=model_mem_after,
        peak_memory_gb=process_mem,
    )


def run_full_pytorch_benchmark(
    config: BenchmarkConfig,
    ckpt_dir: str,
) -> BenchmarkResult:
    """Run full PyTorch/MPS benchmark with actual model weights."""
    try:
        import torch
        from wan.modules.model import WanModel
    except ImportError as e:
        return BenchmarkResult(
            config=config,
            time_per_step_sec=0.0,
            total_time_sec=0.0,
            memory_gb=0.0,
            model_memory_gb=0.0,
            peak_memory_gb=0.0,
            error=f"PyTorch not available: {e}"
        )
    
    logging.info(f"Full PyTorch: frames={config.frames}, steps={config.steps}")
    
    # Determine device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        device_type = 'mps'
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        device_type = 'cuda'
    else:
        device = torch.device('cpu')
        device_type = 'cpu'
    
    logging.info(f"  Using device: {device_type}")
    
    # Load model
    model_path = Path(ckpt_dir) / 'low_noise_model'
    if not model_path.exists():
        return BenchmarkResult(
            config=config,
            time_per_step_sec=0.0,
            total_time_sec=0.0,
            memory_gb=0.0,
            model_memory_gb=0.0,
            peak_memory_gb=0.0,
            error=f"Model not found at {model_path}"
        )
    
    load_start = time.perf_counter()
    model = WanModel.from_pretrained(
        ckpt_dir,
        subfolder='low_noise_model',
        torch_dtype=torch.float32,
    )
    model.eval().requires_grad_(False)
    model.to(device)
    load_time = time.perf_counter() - load_start
    logging.info(f"  Model loaded in {load_time:.1f}s")
    
    model_mem = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**3)
    logging.info(f"  Model memory: {model_mem:.1f}GB")
    
    # Create synthetic inputs
    lat_h, lat_w = 30, 52
    lat_f = (config.frames - 1) // 4 + 1
    seq_len = lat_f * lat_h * lat_w // 4
    in_dim = 16
    
    torch.manual_seed(42)
    x = [torch.randn(in_dim, lat_f, lat_h, lat_w, device=device, dtype=torch.float32)]
    y = [torch.randn(in_dim, lat_f, lat_h, lat_w, device=device, dtype=torch.float32)]
    context = [torch.randn(256, 4096, device=device, dtype=torch.float32)]
    
    step_times: List[float] = []
    
    # Warmup
    with torch.no_grad():
        for _ in range(config.warmup_runs):
            t = torch.tensor([0.5], device=device)
            _ = model(x, t=t, context=context, seq_len=seq_len, y=y)
            if device_type == 'mps':
                torch.mps.synchronize()
            elif device_type == 'cuda':
                torch.cuda.synchronize()
    
    # Benchmark
    gc.collect()
    if device_type == 'mps':
        torch.mps.empty_cache()
    elif device_type == 'cuda':
        torch.cuda.empty_cache()
    
    total_start = time.perf_counter()
    with torch.no_grad():
        for step in range(config.steps):
            timestep = (config.steps - step) / config.steps
            t = torch.tensor([timestep], device=device)
            
            step_start = time.perf_counter()
            output = model(x, t=t, context=context, seq_len=seq_len, y=y)
            if device_type == 'mps':
                torch.mps.synchronize()
            elif device_type == 'cuda':
                torch.cuda.synchronize()
            step_times.append(time.perf_counter() - step_start)
    
    total_time = time.perf_counter() - total_start
    avg_step_time = float(np.mean(step_times))
    
    process_mem = get_process_memory_gb()
    
    # Cleanup
    del model
    gc.collect()
    if device_type == 'mps':
        torch.mps.empty_cache()
    elif device_type == 'cuda':
        torch.cuda.empty_cache()
    
    return BenchmarkResult(
        config=config,
        time_per_step_sec=avg_step_time,
        total_time_sec=total_time,
        memory_gb=process_mem,
        model_memory_gb=model_mem,
        peak_memory_gb=process_mem,
    )


# =============================================================================
# Report Generation
# =============================================================================

def format_time(seconds: float) -> str:
    """Format time in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = seconds / 60
        return f"{mins:.1f}min"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_memory(gb: float) -> str:
    """Format memory in human-readable format."""
    if gb < 1:
        return f"{gb * 1024:.0f}MB"
    else:
        return f"{gb:.1f}GB"


def calculate_speedups(results: List[BenchmarkResult]) -> List[Tuple[BenchmarkResult, float]]:
    """Calculate speedup vs MPS/PyTorch baseline for each config."""
    # Find baselines (PyTorch results)
    baselines: Dict[Tuple[int, int], float] = {}
    for r in results:
        if r.config.backend == 'pytorch' and r.error is None:
            key = (r.config.frames, r.config.steps)
            baselines[key] = r.total_time_sec
    
    # Calculate speedups
    results_with_speedup: List[Tuple[BenchmarkResult, float]] = []
    for r in results:
        key = (r.config.frames, r.config.steps)
        baseline = baselines.get(key)
        if baseline and r.error is None and r.total_time_sec > 0:
            speedup = baseline / r.total_time_sec
        else:
            speedup = 1.0
        results_with_speedup.append((r, speedup))
    
    return results_with_speedup


def generate_markdown_report(suite: BenchmarkSuite, output_path: Path) -> str:
    """Generate markdown report from benchmark results."""
    results_with_speedup = calculate_speedups(suite.results)
    
    lines = [
        "# MLX Backend Benchmarks",
        "",
        "> Auto-generated by `scripts/benchmark_mlx.py`",
        f"> Generated: {suite.timestamp}",
        "",
        "## Configuration",
        "",
        f"- **Hardware**: {suite.hardware}",
        f"- **Model**: {suite.model_info}",
        f"- **Resolution**: {suite.resolution}",
        "",
        "## Results Summary",
        "",
        "| Backend | Frames | Steps | Quant | Time/Step | Total | Model Mem | Speedup |",
        "|---------|--------|-------|-------|-----------|-------|-----------|---------|",
    ]
    
    for r, speedup in results_with_speedup:
        if r.error:
            lines.append(
                f"| {r.config.backend.upper()} | {r.config.frames} | {r.config.steps} | "
                f"{r.config.quant or 'None'} | - | - | - | Error: {r.error[:20]}... |"
            )
        else:
            speedup_str = f"{speedup:.2f}x" if r.config.backend == 'mlx' else "1.00x (baseline)"
            lines.append(
                f"| {r.config.backend.upper()} | {r.config.frames} | {r.config.steps} | "
                f"{r.config.quant or 'None'} | {format_time(r.time_per_step_sec)} | "
                f"{format_time(r.total_time_sec)} | {format_memory(r.model_memory_gb)} | "
                f"{speedup_str} |"
            )
    
    lines.extend([
        "",
        "## Detailed Analysis",
        "",
        "### Quantization Memory Savings",
        "",
    ])
    
    # Group by quant type for memory analysis
    quant_results: Dict[Optional[str], List[BenchmarkResult]] = {}
    for r, _ in results_with_speedup:
        if r.config.backend == 'mlx' and r.error is None:
            if r.config.quant not in quant_results:
                quant_results[r.config.quant] = []
            quant_results[r.config.quant].append(r)
    
    if quant_results:
        lines.append("| Quantization | Avg Model Memory | Memory Reduction |")
        lines.append("|--------------|------------------|------------------|")
        
        baseline_mem: Optional[float] = None
        for quant in [None, 'int8', 'int4', 'nf4']:
            if quant in quant_results:
                avg_mem = float(np.mean([r.model_memory_gb for r in quant_results[quant]]))
                if baseline_mem is None:
                    baseline_mem = avg_mem
                    reduction = "-"
                else:
                    reduction = f"{(1 - avg_mem/baseline_mem) * 100:.0f}%"
                lines.append(f"| {quant or 'None (FP32)'} | {format_memory(avg_mem)} | {reduction} |")
    
    lines.extend([
        "",
        "### Performance by Frame Count",
        "",
    ])
    
    # Group by frame count
    frame_results: Dict[int, List[Tuple[BenchmarkResult, float]]] = {}
    for r, speedup in results_with_speedup:
        if r.error is None:
            if r.config.frames not in frame_results:
                frame_results[r.config.frames] = []
            frame_results[r.config.frames].append((r, speedup))
    
    for frames in sorted(frame_results.keys()):
        results_for_frame = frame_results[frames]
        lines.append(f"**{frames} Frames:**")
        lines.append("")
        for r, speedup in results_for_frame:
            backend = r.config.backend.upper()
            quant = r.config.quant or 'None'
            lines.append(
                f"- {backend} ({quant}): {format_time(r.time_per_step_sec)}/step, "
                f"{format_time(r.total_time_sec)} total, {speedup:.2f}x speedup"
            )
        lines.append("")
    
    lines.extend([
        "## Methodology",
        "",
        "### Benchmark Procedure",
        "",
        "1. **Warmup**: 1-2 forward passes to trigger JIT compilation",
        "2. **Timing**: Each denoising step is timed individually",
        "3. **Synchronization**: GPU operations are synchronized before timing",
        "4. **Memory**: Peak process memory is recorded",
        "",
        "### Notes",
        "",
        "- All timings exclude model loading and VAE encode/decode",
        "- Memory figures are for a single DiT model (full pipeline uses 2x)",
        "- Synthetic benchmarks use smaller models; full benchmarks use production weights",
        "- MPS backend uses float32 (bfloat16 not supported)",
        "",
        "## Running Benchmarks",
        "",
        "```bash",
        "# Quick synthetic benchmark (no model required)",
        "python scripts/benchmark_mlx.py --mode synthetic",
        "",
        "# Full benchmark with model weights",
        "python scripts/benchmark_mlx.py --ckpt_dir models/lingbot-world-base-cam",
        "",
        "# Specific configuration",
        "python scripts/benchmark_mlx.py --frames 41 --steps 20 --quant nf4",
        "```",
        "",
    ])
    
    content = "\n".join(lines)
    
    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(content)
    
    logging.info(f"Report written to {output_path}")
    
    return content


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description='MLX Backend Benchmarks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--mode', choices=['synthetic', 'full'], default='synthetic',
        help='Benchmark mode: synthetic (no model) or full (requires weights)'
    )
    parser.add_argument(
        '--ckpt_dir', default='models/lingbot-world-base-cam',
        help='Model checkpoint directory (for full mode)'
    )
    parser.add_argument(
        '--frames', type=int, nargs='+', default=None,
        help='Frame counts to benchmark (default: 5, 41)'
    )
    parser.add_argument(
        '--steps', type=int, nargs='+', default=None,
        help='Step counts to benchmark (default: 5, 20)'
    )
    parser.add_argument(
        '--quant', type=str, nargs='+', default=None,
        help='Quantization types (none, int8, int4, nf4)'
    )
    parser.add_argument(
        '--backends', type=str, nargs='+', default=['pytorch', 'mlx'],
        help='Backends to benchmark'
    )
    parser.add_argument(
        '--output', type=str, default='docs/MLX_BENCHMARKS.md',
        help='Output markdown file path'
    )
    parser.add_argument(
        '--warmup', type=int, default=1,
        help='Number of warmup runs before timing'
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Default configurations
    if args.frames is None:
        args.frames = [5, 41]
    if args.steps is None:
        args.steps = [5, 20]
    if args.quant is None:
        args.quant = ['none', 'int8', 'nf4']
    
    # Build benchmark configurations
    configs: List[BenchmarkConfig] = []
    
    for frames in args.frames:
        # Match steps to frames (quick test vs production)
        steps = 5 if frames <= 5 else 20
        if len(args.steps) == 1:
            steps = args.steps[0]
        elif frames <= 5 and 5 in args.steps:
            steps = 5
        elif frames > 5 and 20 in args.steps:
            steps = 20
        else:
            steps = args.steps[0]
        
        for backend in args.backends:
            # PyTorch doesn't support our quantization
            quant_list = ['none'] if backend == 'pytorch' else args.quant
            
            for quant in quant_list:
                configs.append(BenchmarkConfig(
                    frames=frames,
                    steps=steps,
                    quant=quant if quant != 'none' else None,
                    backend=backend,  # type: ignore
                    warmup_runs=args.warmup,
                ))
    
    logging.info(f"Running {len(configs)} benchmark configurations")
    logging.info(f"Mode: {args.mode}")
    logging.info(f"Hardware: {get_hardware_info()}")
    
    # Initialize suite
    suite = BenchmarkSuite(
        hardware=get_hardware_info(),
        model_info="WanModel (14B parameters)" if args.mode == 'full' else "WanModel (synthetic, ~100M)",
        resolution="480x832",
        timestamp=datetime.now().isoformat(),
    )
    
    # Run benchmarks
    for i, config in enumerate(configs):
        logging.info(f"\n[{i+1}/{len(configs)}] Running: {config.backend}/{config.frames}f/{config.steps}s/{config.quant or 'none'}")
        
        try:
            if args.mode == 'synthetic':
                if config.backend == 'mlx':
                    result = run_synthetic_mlx_benchmark(config)
                else:
                    result = run_synthetic_pytorch_benchmark(config)
            else:
                if config.backend == 'mlx':
                    result = run_full_mlx_benchmark(config, args.ckpt_dir)
                else:
                    result = run_full_pytorch_benchmark(config, args.ckpt_dir)
            
            suite.results.append(result)
            
            if result.error:
                logging.warning(f"  Error: {result.error}")
            else:
                logging.info(
                    f"  Result: {format_time(result.time_per_step_sec)}/step, "
                    f"{format_time(result.total_time_sec)} total, "
                    f"{format_memory(result.model_memory_gb)} model"
                )
        
        except Exception as e:
            logging.error(f"  Exception: {e}")
            suite.results.append(BenchmarkResult(
                config=config,
                time_per_step_sec=0.0,
                total_time_sec=0.0,
                memory_gb=0.0,
                model_memory_gb=0.0,
                peak_memory_gb=0.0,
                error=str(e),
            ))
        
        # GC between runs
        gc.collect()
    
    # Generate report
    output_path = Path(args.output)
    report = generate_markdown_report(suite, output_path)
    
    # Print summary to console
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    
    results_with_speedup = calculate_speedups(suite.results)
    for r, speedup in results_with_speedup:
        if r.error is None:
            print(
                f"{r.config.backend:>8} | {r.config.frames:>3}f | {r.config.steps:>2}s | "
                f"{r.config.quant or 'none':>5} | "
                f"{format_time(r.time_per_step_sec):>8}/step | "
                f"{format_time(r.total_time_sec):>8} | "
                f"{speedup:.2f}x"
            )
    
    print("\n" + f"Full report: {output_path}")


if __name__ == '__main__':
    main()
