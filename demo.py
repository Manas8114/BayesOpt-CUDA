import argparse
import time
import torch
import numpy as np
import scipy.linalg

from bayesopt_cuda.kernels_optimized import (
    rbf_kernel_tiled,
    rbf_kernel_tiled_fp16,
    acquisition_fused_chol
)

def main():
    parser = argparse.ArgumentParser(description="BayesOpt-CUDA Live Demo")
    parser.add_argument("--n", type=int, default=1024, help="Number of training points")
    parser.add_argument("--m", type=int, default=4096, help="Number of test points")
    parser.add_argument("--d", type=int, default=16, help="Dimensionality")
    args = parser.parse_args()

    print(f"==================================================")
    print(f" BayesOpt-CUDA Live Demo")
    print(f"==================================================")
    print(f"Dataset:")
    print(f"  Training points (n): {args.n}")
    print(f"  Candidate points (m): {args.m}")
    print(f"  Dimensions (d): {args.d}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"==================================================\n")

    device = "cuda"
    dtype = torch.float32

    # 1. Generate Synthetic Data
    print("[1/3] Generating synthetic data...")
    X_train = torch.randn(args.n, args.d, dtype=dtype, device=device)
    X_test  = torch.randn(args.m, args.d, dtype=dtype, device=device)
    Y_train = torch.randn(args.n, dtype=dtype, device=device)

    ls, var = 1.0, 1.0
    
    # 2. Build Covariance Matrix (RBF Kernel)
    print("[2/3] Constructing Covariance Matrix (FP16 Mixed Precision)...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    
    # Use FP16 kernel for speed
    X_train_h = X_train.half()
    K_train = rbf_kernel_tiled_fp16(X_train_h, X_train_h, ls, var)
    
    # Add jitter for numerical stability
    K_train += torch.eye(args.n, device=device) * 1e-4
    
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    print(f"      Done in {(t1-t0)*1000:.2f} ms")

    # 3. Fit GP and Evaluate Acquisition
    print("[3/3] Fitting GP and evaluating Expected Improvement (EI)...")
    
    # GP Fitting (Cholesky factorization is O(n^3), done via highly optimized cuBLAS)
    t0 = time.perf_counter()
    L = torch.linalg.cholesky(K_train)
    alpha = torch.cholesky_solve(Y_train.unsqueeze(-1), L).squeeze(-1)
    torch.cuda.synchronize()
    t_fit = time.perf_counter() - t0

    # Fused Acquisition Function
    t0 = time.perf_counter()
    y_best = Y_train.max().item()
    ei_values = acquisition_fused_chol(
        X_test, X_train, alpha, L,
        lengthscale=ls, variance=var, y_best=y_best, exploration_param=0.01, acq_type="ei"
    )
    torch.cuda.synchronize()
    t_acq = time.perf_counter() - t0

    print(f"      GP Fit (cuBLAS L):  {t_fit*1000:.2f} ms")
    print(f"      EI Eval (custom):   {t_acq*1000:.2f} ms")

    # Results
    best_idx = torch.argmax(ei_values).item()
    best_ei = ei_values[best_idx].item()
    print(f"\n==================================================")
    print(f" => Best candidate index: {best_idx}")
    print(f" => Expected Improvement: {best_ei:.4f}")
    print(f"==================================================")


if __name__ == "__main__":
    main()
