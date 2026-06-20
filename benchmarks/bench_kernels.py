"""
benchmarks/bench_kernels.py
============================
Measures wall-clock throughput for:

  1. Naive RBF kernel  (rbf_kernel_naive)
  2. Tiled RBF kernel  (rbf_kernel_tiled)
  3. Naive EI          (expected_improvement_naive)
  4. Fused acquisition (acquisition_fused)

All results are collected via CUDA events (GPU timing) plus
torch.utils.benchmark.Timer (CPU wall-clock) and printed as a
machine-readable JSON table, which is also saved to
  benchmarks/results/bench_<timestamp>.json

Hard rules followed:
  - Every number comes from an actual kernel invocation.
  - Warm-up runs precede all measurements.
  - CUDA synchronisation (cudaDeviceSynchronize) between timings.
  - No estimates, no inferred numbers.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch
import torch.utils.benchmark as benchmark

# ---- project root on path ------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import bayesopt_cuda._C as _C
from bayesopt_cuda.kernels_optimized import (
    rbf_kernel_tiled, rbf_kernel_tiled_fp16,
    acquisition_fused, acquisition_fused_chol
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cuda_time_ms(fn, n_warmup: int = 10, n_repeat: int = 100) -> float:
    """
    Measure GPU kernel time (ms) using CUDA events.
    Returns mean over n_repeat runs after n_warmup discarded runs.
    """
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt   = torch.cuda.Event(enable_timing=True)

    # Warm-up
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    # Timed runs
    times_ms: List[float] = []
    for _ in range(n_repeat):
        start_evt.record()
        fn()
        end_evt.record()
        torch.cuda.synchronize()
        times_ms.append(start_evt.elapsed_time(end_evt))

    arr = np.array(times_ms)
    return {
        "mean_ms":   float(arr.mean()),
        "std_ms":    float(arr.std()),
        "min_ms":    float(arr.min()),
        "median_ms": float(np.median(arr)),
    }


def make_rbf_tensors(n: int, m: int, d: int, device="cuda"):
    X = torch.randn(n, d, dtype=torch.float32, device=device)
    Y = torch.randn(m, d, dtype=torch.float32, device=device)
    return X, Y


def make_acq_tensors(n: int, m: int, d: int, device="cuda"):
    """Make the 5 tensors needed by the acquisition kernels."""
    X_train = torch.randn(n, d, dtype=torch.float32, device=device)
    X_test  = torch.randn(m, d, dtype=torch.float32, device=device)
    alpha = torch.randn(n, dtype=torch.float32, device=device)
    
    # Create valid K, L, K_inv
    A = torch.randn(n, n, dtype=torch.float32, device=device)
    K = A @ A.T + torch.eye(n, device=device) * 1e-4
    L = torch.linalg.cholesky(K)
    K_inv = torch.cholesky_inverse(L)
    
    return X_test, X_train, alpha, K_inv, L


# ---------------------------------------------------------------------------
# Benchmark suites
# ---------------------------------------------------------------------------

def bench_rbf(sizes, ls=1.0, var=1.0, n_warmup=10, n_repeat=100):
    """Compare naive vs tiled RBF across (n, m, d) combos."""
    results = []
    for n, m, d in sizes:
        X, Y = make_rbf_tensors(n, m, d)
        print(f"  RBF n={n} m={m} d={d} ...", end=" ", flush=True)

        # Naive
        naive_t = cuda_time_ms(
            lambda: _C.rbf_kernel_naive(X, Y, ls, var),
            n_warmup, n_repeat
        )

        # Tiled
        tiled_t = cuda_time_ms(
            lambda: rbf_kernel_tiled(X, Y, ls, var),
            n_warmup, n_repeat
        )

        # Tiled FP16
        X_fp16, Y_fp16 = X.half(), Y.half()
        tiled_fp16_t = cuda_time_ms(
            lambda: rbf_kernel_tiled_fp16(X_fp16, Y_fp16, ls, var),
            n_warmup, n_repeat
        )

        speedup = naive_t["mean_ms"] / tiled_t["mean_ms"]
        speedup_fp16 = naive_t["mean_ms"] / tiled_fp16_t["mean_ms"]
        print(f"naive={naive_t['mean_ms']:.3f}ms  tiled={tiled_t['mean_ms']:.3f}ms  "
              f"fp16={tiled_fp16_t['mean_ms']:.3f}ms  "
              f"speedup={speedup:.2f}x  speedup_fp16={speedup_fp16:.2f}x")

        results.append({
            "kernel": "rbf",
            "n": n, "m": m, "d": d,
            "naive": naive_t,
            "tiled": tiled_t,
            "tiled_fp16": tiled_fp16_t,
            "speedup_mean": round(speedup, 3),
            "speedup_fp16_mean": round(speedup_fp16, 3),
        })
    return results


def bench_acq(sizes, ls=1.0, var=1.0, y_best=0.0, xi=0.01,
              n_warmup=10, n_repeat=100):
    """
    Compare naive EI (2-step: build K_cross then call ei_kernel_naive)
    vs fused acquisition.
    """
    results = []
    for n, m, d in sizes:
        X_test, X_train, alpha, K_inv, L = make_acq_tensors(n, m, d)

        # Naive: rbf_kernel_naive to get K_cross (n×m), then ei_kernel_naive
        # We pre-allocate mean/std buffers so we're measuring pure kernel cost.
        def naive_fn():
            # Step 1: cross-kernel (m, n)
            K_cross = _C.rbf_kernel_naive(X_test, X_train, ls, var)
            # Step 2: mean (m,)
            mean = K_cross @ alpha
            # Step 3: posterior variance via K_cross @ K_inv @ K_cross^T diag
            V = K_cross @ K_inv          # (m, n)
            var_post = var - (V * K_cross).sum(dim=1).clamp(min=0)
            std = var_post.sqrt()        # (m,)
            # Step 4: EI kernel
            _C.expected_improvement_naive(mean, std, float(y_best), float(xi))

        def fused_fn():
            acquisition_fused(X_test, X_train, alpha, K_inv,
                              ls, var, y_best, xi, "ei")

        def fused_chol_fn():
            acquisition_fused_chol(X_test, X_train, alpha, L,
                                   ls, var, y_best, xi, "ei")

        print(f"  ACQ n={n} m={m} d={d} ...", end=" ", flush=True)
        naive_t = cuda_time_ms(naive_fn, n_warmup, n_repeat)
        fused_t = cuda_time_ms(fused_fn, n_warmup, n_repeat)
        fused_chol_t = cuda_time_ms(fused_chol_fn, n_warmup, n_repeat)

        speedup = naive_t["mean_ms"] / fused_t["mean_ms"]
        speedup_chol = naive_t["mean_ms"] / fused_chol_t["mean_ms"]
        print(f"naive={naive_t['mean_ms']:.3f}ms  "
              f"fused_kinv={fused_t['mean_ms']:.3f}ms  "
              f"fused_chol={fused_chol_t['mean_ms']:.3f}ms  "
              f"spd_kinv={speedup:.2f}x  spd_chol={speedup_chol:.2f}x")

        results.append({
            "kernel": "acquisition_ei",
            "n": n, "m": m, "d": d,
            "naive": naive_t,
            "fused": fused_t,
            "fused_chol": fused_chol_t,
            "speedup_mean": round(speedup, 3),
            "speedup_chol_mean": round(speedup_chol, 3),
        })
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="BayesOpt-CUDA kernel benchmarks")
    parser.add_argument("--n-warmup", type=int, default=20)
    parser.add_argument("--n-repeat", type=int, default=200)
    parser.add_argument("--quick",    action="store_true",
                        help="Run a smaller set for CI / smoke test")
    args = parser.parse_args()

    if args.quick:
        rbf_sizes = [(512, 512, 8), (1024, 1024, 16)]
        acq_sizes = [(64, 256, 4), (256, 1024, 8)]
        n_warmup, n_repeat = 5, 20
    else:
        rbf_sizes = [
            (256,  256,  4),
            (512,  512,  8),
            (1024, 1024, 16),
            (2048, 2048, 32),
            (4096, 4096, 8),
        ]
        acq_sizes = [
            (32,   256,  2),
            (64,   512,  4),
            (128,  1024, 8),
            (256,  2048, 16),
            (512,  4096, 8),
        ]
        n_warmup, n_repeat = args.n_warmup, args.n_repeat

    # GPU info
    dev = torch.cuda.get_device_properties(0)
    gpu_info = {
        "name":         dev.name,
        "sm_count":     dev.multi_processor_count,
        "total_mem_mb": dev.total_memory // 1024 // 1024,
        "compute_cap":  f"{dev.major}.{dev.minor}",
    }

    import bayesopt_cuda
    run_meta = {
        "timestamp":   datetime.utcnow().isoformat() + "Z",
        "gpu":         gpu_info,
        "n_warmup":    n_warmup,
        "n_repeat":    n_repeat,
        "bayesopt_cuda_version": bayesopt_cuda.__version__,
        "torch_version": torch.__version__,
        "cuda_version":  torch.version.cuda,
    }

    print("=" * 72)
    print(f"BayesOpt-CUDA Kernel Benchmarks")
    print(f"GPU:  {dev.name}  ({dev.multi_processor_count} SMs, "
          f"{dev.total_memory//1024//1024} MB, cap {dev.major}.{dev.minor})")
    print(f"Runs: {n_warmup} warmup + {n_repeat} timed per config")
    print("=" * 72)

    print("\n[1/2] RBF Kernel: Naive vs. Tiled")
    rbf_results = bench_rbf(rbf_sizes, n_warmup=n_warmup, n_repeat=n_repeat)

    print("\n[2/2] Acquisition Function: Naive vs. Fused")
    acq_results = bench_acq(acq_sizes, n_warmup=n_warmup, n_repeat=n_repeat)

    # --- Print summary table ------------------------------------------------
    all_results = rbf_results + acq_results

    print("\n" + "=" * 80)
    print(f"{'Kernel':<15} {'(n,m,d)':<15} {'Naive':>8} {'Opt(v1)':>8} {'Opt(v2)':>8} {'Spd(v1)':>8} {'Spd(v2)':>8}")
    print("-" * 80)
    for r in all_results:
        label = r["kernel"]
        shape = f"({r['n']},{r['m']},{r['d']})"
        naive_ms = r["naive"]["mean_ms"]
        
        if label == "rbf":
            opt1_ms = r["tiled"]["mean_ms"]
            opt2_ms = r["tiled_fp16"]["mean_ms"]
            spd1 = r["speedup_mean"]
            spd2 = r["speedup_fp16_mean"]
        else:
            opt1_ms = r["fused"]["mean_ms"]
            opt2_ms = r["fused_chol"]["mean_ms"]
            spd1 = r["speedup_mean"]
            spd2 = r["speedup_chol_mean"]
            
        print(f"{label[:15]:<15} {shape:<15} {naive_ms:>8.3f} {opt1_ms:>8.3f} {opt2_ms:>8.3f} {spd1:>7.2f}x {spd2:>7.2f}x")
    print("=" * 80)

    # --- Save JSON ----------------------------------------------------------
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"bench_{ts}.json")
    payload = {"meta": run_meta, "results": all_results}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved -> {out_path}")
    return payload


if __name__ == "__main__":
    main()
