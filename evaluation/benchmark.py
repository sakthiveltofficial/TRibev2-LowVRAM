"""
benchmark.py — GPU Memory & Inference Performance Benchmarking
==============================================================
Profiles VRAM consumption, latency, and throughput on RTX 5060 Ti (8GB)
or CPU environments for TRIBE-local components.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("tribe_local.benchmark")


class VRAMBenchmark:
    """Measures peak VRAM allocation and inference latency for PyTorch models."""

    @staticmethod
    def get_gpu_info() -> Dict[str, Any]:
        """Query connected GPU properties."""
        if not torch.cuda.is_available():
            return {"cuda_available": False, "device_name": "CPU"}

        props = torch.cuda.get_device_properties(0)
        return {
            "cuda_available": True,
            "device_name": props.name,
            "total_vram_mb": props.total_memory / (1024**2),
            "total_vram_gb": props.total_memory / (1024**3),
            "cuda_version": torch.version.cuda,
            "major_minor": f"{props.major}.{props.minor}",
        }

    @staticmethod
    def count_parameters(model: nn.Module) -> Dict[str, int]:
        """Count total, trainable, and frozen parameters."""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = total - trainable
        return {
            "total_params": total,
            "trainable_params": trainable,
            "frozen_params": frozen,
            "total_param_mb": (total * 4) / (1024**2),  # FP32 basis
        }

    @classmethod
    def measure_forward_vram(
        cls,
        model: nn.Module,
        dummy_inputs: Any,
        device: str = "cuda",
        fp16: bool = True,
    ) -> Dict[str, float]:
        """Measure peak GPU memory consumed during a single forward pass."""
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        model = model.to(device)
        model.eval()

        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            baseline_mem = torch.cuda.memory_allocated() / (1024**2)

            with torch.inference_mode():
                with torch.cuda.amp.autocast(enabled=fp16):
                    if isinstance(dummy_inputs, dict):
                        _ = model(**dummy_inputs)
                    elif isinstance(dummy_inputs, (tuple, list)):
                        _ = model(*dummy_inputs)
                    else:
                        _ = model(dummy_inputs)

            peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
            reserved_mem = torch.cuda.max_memory_reserved() / (1024**2)

            return {
                "baseline_vram_mb": round(baseline_mem, 2),
                "peak_vram_mb": round(peak_mem, 2),
                "delta_vram_mb": round(peak_mem - baseline_mem, 2),
                "reserved_vram_mb": round(reserved_mem, 2),
            }
        else:
            # CPU fallback: time execution only
            with torch.inference_mode():
                if isinstance(dummy_inputs, dict):
                    _ = model(**dummy_inputs)
                elif isinstance(dummy_inputs, (tuple, list)):
                    _ = model(*dummy_inputs)
                else:
                    _ = model(dummy_inputs)
            return {"baseline_vram_mb": 0.0, "peak_vram_mb": 0.0, "delta_vram_mb": 0.0}

    @classmethod
    def measure_latency(
        cls,
        model: nn.Module,
        dummy_inputs: Any,
        device: str = "cuda",
        fp16: bool = True,
        n_warmup: int = 3,
        n_runs: int = 20,
    ) -> Dict[str, float]:
        """Measure mean, median, min, max latency over repeated runs."""
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        model = model.to(device)
        model.eval()

        # Warmup
        with torch.inference_mode():
            for _ in range(n_warmup):
                with torch.cuda.amp.autocast(enabled=(device == "cuda" and fp16)):
                    if isinstance(dummy_inputs, dict):
                        _ = model(**dummy_inputs)
                    elif isinstance(dummy_inputs, (tuple, list)):
                        _ = model(*dummy_inputs)
                    else:
                        _ = model(dummy_inputs)
        if device == "cuda":
            torch.cuda.synchronize()

        # Timed runs
        timings: List[float] = []
        with torch.inference_mode():
            for _ in range(n_runs):
                t0 = time.perf_counter()
                with torch.cuda.amp.autocast(enabled=(device == "cuda" and fp16)):
                    if isinstance(dummy_inputs, dict):
                        _ = model(**dummy_inputs)
                    elif isinstance(dummy_inputs, (tuple, list)):
                        _ = model(*dummy_inputs)
                    else:
                        _ = model(dummy_inputs)
                if device == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                timings.append((t1 - t0) * 1000.0)  # ms

        timings_arr = np.array(timings)
        return {
            "mean_latency_ms": round(float(np.mean(timings_arr)), 2),
            "median_latency_ms": round(float(np.median(timings_arr)), 2),
            "min_latency_ms": round(float(np.min(timings_arr)), 2),
            "max_latency_ms": round(float(np.max(timings_arr)), 2),
            "std_latency_ms": round(float(np.std(timings_arr)), 2),
            "throughput_samples_per_sec": round(1000.0 / float(np.mean(timings_arr)), 2),
        }


def run_benchmark_suite(
    output_dir: Optional[str] = "benchmarks",
    fp16: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    """Execute a comprehensive benchmark covering all pipeline components."""
    from ..brain.subject_layers import BrainPredictionHead
    from ..fusion.cross_attention_fusion import TribeLocalFusion

    out_p = Path(output_dir) if output_dir else Path("benchmarks")
    out_p.mkdir(parents=True, exist_ok=True)

    gpu_info = VRAMBenchmark.get_gpu_info()
    logger.info(f"Running benchmark on: {gpu_info.get('device_name')}")

    results: Dict[str, Any] = {"gpu_info": gpu_info, "fp16": fp16, "device": device}

    # 1. Benchmark Fusion Module Alone
    logger.info("Benchmarking TribeLocalFusion (Transformer + Cross-Attention)...")
    fusion = TribeLocalFusion(
        encoder_dims={"video": 384, "audio": 768, "text": 1536},
        hidden=512,
        low_rank=1024,
        num_transformer_layers=4,
    )
    fusion_params = VRAMBenchmark.count_parameters(fusion)

    dummy_features = {
        "video": torch.randn(1, 384, 200, device=device),
        "audio": torch.randn(1, 768, 200, device=device),
        "text": torch.randn(1, 1536, 200, device=device),
    }

    fusion_vram = VRAMBenchmark.measure_forward_vram(fusion, dummy_features, device=device, fp16=fp16)
    fusion_latency = VRAMBenchmark.measure_latency(fusion, dummy_features, device=device, fp16=fp16)

    results["fusion"] = {
        "parameters": fusion_params,
        "vram": fusion_vram,
        "latency": fusion_latency,
    }

    # 2. Benchmark Brain Head (Voxel Mode: 20484 vertices)
    logger.info("Benchmarking BrainPredictionHead (Voxel Mode: 20484 vertices)...")
    voxel_head = BrainPredictionHead(mode="voxel", r=1024, n_vertices=20484, n_subjects=25, target_trs=100)
    voxel_params = VRAMBenchmark.count_parameters(voxel_head)

    dummy_bottleneck = torch.randn(1, 1024, 200, device=device)
    dummy_subjs = torch.tensor([0], device=device)

    voxel_vram = VRAMBenchmark.measure_forward_vram(voxel_head, (dummy_bottleneck, dummy_subjs), device=device, fp16=fp16)
    voxel_latency = VRAMBenchmark.measure_latency(voxel_head, (dummy_bottleneck, dummy_subjs), device=device, fp16=fp16)

    results["voxel_head"] = {
        "parameters": voxel_params,
        "vram": voxel_vram,
        "latency": voxel_latency,
    }

    # 3. Benchmark Brain Head (ROI Mode: 360 ROIs)
    logger.info("Benchmarking BrainPredictionHead (ROI Mode: 360 ROIs)...")
    roi_head = BrainPredictionHead(mode="roi", r=1024, n_rois=360, n_subjects=25, target_trs=100)
    roi_params = VRAMBenchmark.count_parameters(roi_head)

    roi_vram = VRAMBenchmark.measure_forward_vram(roi_head, (dummy_bottleneck, dummy_subjs), device=device, fp16=fp16)
    roi_latency = VRAMBenchmark.measure_latency(roi_head, (dummy_bottleneck, dummy_subjs), device=device, fp16=fp16)

    results["roi_head"] = {
        "parameters": roi_params,
        "vram": roi_vram,
        "latency": roi_latency,
    }

    # 4. Benchmark Full Integrated Forward Pass
    logger.info("Benchmarking Integrated Pipeline (Fusion + Voxel Head)...")

    class IntegratedModel(nn.Module):
        def __init__(self, f_mod, h_mod):
            super().__init__()
            self.fusion = f_mod
            self.head = h_mod

        def forward(self, feats, s_ids):
            z = self.fusion(feats)
            return self.head(z, s_ids)

    integrated = IntegratedModel(fusion, voxel_head)
    integrated_params = VRAMBenchmark.count_parameters(integrated)
    integrated_vram = VRAMBenchmark.measure_forward_vram(integrated, (dummy_features, dummy_subjs), device=device, fp16=fp16)
    integrated_latency = VRAMBenchmark.measure_latency(integrated, (dummy_features, dummy_subjs), device=device, fp16=fp16)

    results["integrated_pipeline"] = {
        "parameters": integrated_params,
        "vram": integrated_vram,
        "latency": integrated_latency,
    }

    # 5. Check 8GB VRAM Compliance
    peak_mb = integrated_vram.get("peak_vram_mb", 0.0)
    limit_mb = 8192.0  # 8GB VRAM
    vram_safe = peak_mb < 7000.0  # 7GB safe margin

    results["vram_budget_check"] = {
        "target_hardware": "RTX 5060 Ti (8GB)",
        "vram_limit_mb": limit_mb,
        "peak_used_mb": peak_mb,
        "within_8gb_budget": peak_mb < limit_mb,
        "safe_headroom_mb": round(limit_mb - peak_mb, 2),
        "status": "PASS (Fits comfortably in 8GB)" if vram_safe else "WARNING: High VRAM",
    }

    # Save to file
    out_file = out_p / "benchmark_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Benchmark finished! Report written to: {out_file}")
    _print_summary_table(results)

    return results


def _print_summary_table(res: Dict[str, Any]) -> None:
    """Pretty-print benchmark results table."""
    print("\n" + "=" * 76)
    print("  TRIBE-LOCAL BENCHMARK SUMMARY (Target: RTX 5060 Ti 8GB VRAM)")
    print("=" * 76)
    print(f"Device: {res['gpu_info'].get('device_name')} | FP16: {res['fp16']}")
    print("-" * 76)
    fmt = "{:<26} | {:<12} | {:<14} | {:<14}"
    print(fmt.format("Component", "Total Params", "Peak VRAM (MB)", "Latency (ms)"))
    print("-" * 76)

    for comp in ("fusion", "voxel_head", "roi_head", "integrated_pipeline"):
        c_data = res.get(comp, {})
        n_p = f"{c_data.get('parameters', {}).get('total_params', 0):,}"
        vram = f"{c_data.get('vram', {}).get('peak_vram_mb', 0.0):.1f}"
        lat = f"{c_data.get('latency', {}).get('mean_latency_ms', 0.0):.1f}"
        print(fmt.format(comp, n_p, vram, lat))

    print("-" * 76)
    check = res.get("vram_budget_check", {})
    print(f"8GB VRAM Budget Status: {check.get('status')} (Headroom: {check.get('safe_headroom_mb')} MB)")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="benchmarks")
    parser.add_argument("--no-fp16", action="store_true")
    args = parser.parse_args()

    run_benchmark_suite(output_dir=args.output_dir, fp16=not args.no_fp16)
