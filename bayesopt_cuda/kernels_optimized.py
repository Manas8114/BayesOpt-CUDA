"""
Python wrappers for optimised CUDA kernels.

Kernels:
  rbf_kernel_tiled          - Tiled RBF, float32 I/O.
  acquisition_fused         - Fused EI/UCB/PI with K_inv (fast for n<=64).
  acquisition_fused_chol    - Fused EI/UCB/PI with Cholesky L (O(n) SMEM, works for any n).
  rbf_kernel_tiled_fp16     - Tiled RBF, float16 inputs / float32 output.
"""

import torch
from typing import Tuple

def rbf_kernel_tiled(
    X: torch.Tensor,
    Y: torch.Tensor,
    lengthscale: float = 1.0,
    variance: float = 1.0,
) -> torch.Tensor:
    """
    Optimized Tiled CUDA RBF Kernel.

    Args:
        X: (n, d) float32 tensor on CUDA
        Y: (m, d) float32 tensor on CUDA
        lengthscale: RBF lengthscale
        variance: RBF variance

    Returns:
        K: (n, m) covariance matrix
    """
    import bayesopt_cuda._C as _C
    return _C.rbf_kernel_tiled(X.contiguous(), Y.contiguous(), float(lengthscale), float(variance))


def acquisition_fused(
    X_test: torch.Tensor,
    X_train: torch.Tensor,
    alpha: torch.Tensor,
    K_inv: torch.Tensor,
    lengthscale: float = 1.0,
    variance: float = 1.0,
    y_best: float = 0.0,
    exploration_param: float = 0.01,
    acq_type: str = "ei"
) -> torch.Tensor:
    """
    Fused Acquisition Function (EI, UCB, PI).

    Args:
        X_test: (m, d) float32 tensor on CUDA
        X_train: (n, d) float32 tensor on CUDA
        alpha: (n,) float32 tensor on CUDA (K_train^-1 * y_train)
        K_inv: (n, n) float32 tensor on CUDA (K_train^-1)
        lengthscale: RBF lengthscale
        variance: RBF variance
        y_best: Best observed value so far (for EI, PI)
        exploration_param: xi (for EI, PI) or beta (for UCB)
        acq_type: "ei", "ucb", or "pi"

    Returns:
        acq_values: (m,) acquisition function values
    """
    import bayesopt_cuda._C as _C
    
    types = {"ei": 0, "ucb": 1, "pi": 2}
    if acq_type.lower() not in types:
        raise ValueError(f"Unknown acquisition type: {acq_type}")
        
    type_idx = types[acq_type.lower()]

    return _C.acquisition_fused(
        X_test.contiguous(),
        X_train.contiguous(),
        alpha.contiguous(),
        K_inv.contiguous(),
        float(lengthscale),
        float(variance),
        float(y_best),
        float(exploration_param),
        int(type_idx)
    )

# NumPy fallbacks for testing

import numpy as np

def rbf_kernel_tiled_np(
    X: np.ndarray,
    Y: np.ndarray,
    lengthscale: float = 1.0,
    variance: float = 1.0,
) -> np.ndarray:
    X_t = torch.from_numpy(X).cuda()
    Y_t = torch.from_numpy(Y).cuda()
    K_t = rbf_kernel_tiled(X_t, Y_t, lengthscale, variance)
    return K_t.cpu().numpy()

def acquisition_fused_np(
    X_test: np.ndarray,
    X_train: np.ndarray,
    alpha: np.ndarray,
    K_inv: np.ndarray,
    lengthscale: float = 1.0,
    variance: float = 1.0,
    y_best: float = 0.0,
    exploration_param: float = 0.01,
    acq_type: str = "ei"
) -> np.ndarray:
    X_test_t = torch.from_numpy(X_test).cuda()
    X_train_t = torch.from_numpy(X_train).cuda()
    alpha_t = torch.from_numpy(alpha).cuda()
    K_inv_t = torch.from_numpy(K_inv).cuda()
    
    res = acquisition_fused(
        X_test_t, X_train_t, alpha_t, K_inv_t,
        lengthscale, variance, y_best, exploration_param, acq_type
    )
    return res.cpu().numpy()

def acquisition_fused_chol(
    X_test: torch.Tensor,
    X_train: torch.Tensor,
    alpha: torch.Tensor,
    L: torch.Tensor,
    lengthscale: float = 1.0,
    variance: float = 1.0,
    y_best: float = 0.0,
    exploration_param: float = 0.01,
    acq_type: str = "ei"
) -> torch.Tensor:
    """
    Fused acquisition using the Cholesky factor L (lower triangular).

    Replaces K_inv with an in-kernel forward substitution (Lv = k_s),
    reducing shared memory from O(n²) to O(n). Works for any n.

    Args:
        X_test:           (m, d) float32 CUDA
        X_train:          (n, d) float32 CUDA
        alpha:            (n,)   float32 CUDA  -- K_train^{-1} y
        L:                (n, n) float32 CUDA  -- lower-triangular Cholesky of K_train
        lengthscale, variance, y_best, exploration_param, acq_type: same as acquisition_fused.

    Returns:
        acq_values: (m,) float32
    """
    import bayesopt_cuda._C as _C
    types = {"ei": 0, "ucb": 1, "pi": 2}
    if acq_type.lower() not in types:
        raise ValueError(f"Unknown acquisition type: {acq_type}")
    return _C.acquisition_fused_chol(
        X_test.contiguous(), X_train.contiguous(),
        alpha.contiguous(), L.contiguous(),
        float(lengthscale), float(variance),
        float(y_best), float(exploration_param),
        int(types[acq_type.lower()])
    )


def rbf_kernel_tiled_fp16(
    X: torch.Tensor,
    Y: torch.Tensor,
    lengthscale: float = 1.0,
    variance: float = 1.0,
) -> torch.Tensor:
    """
    Tiled RBF kernel accepting float16 inputs and returning float32 output.

    Loads X/Y as FP16 (2× fewer bytes in global memory), computes
    squared distances with __half2 SIMD, accumulates in FP32, and
    returns a float32 kernel matrix.

    Args:
        X: (n, d) float16 CUDA
        Y: (m, d) float16 CUDA
        lengthscale, variance: RBF hyperparameters

    Returns:
        K: (n, m) float32 CUDA
    """
    import bayesopt_cuda._C as _C
    return _C.rbf_kernel_tiled_fp16(
        X.contiguous(), Y.contiguous(),
        float(lengthscale), float(variance)
    )


# NumPy helpers for testing

def acquisition_fused_chol_np(
    X_test: np.ndarray,
    X_train: np.ndarray,
    alpha: np.ndarray,
    L: np.ndarray,
    lengthscale: float = 1.0,
    variance: float = 1.0,
    y_best: float = 0.0,
    exploration_param: float = 0.01,
    acq_type: str = "ei"
) -> np.ndarray:
    res = acquisition_fused_chol(
        torch.from_numpy(X_test).cuda(),
        torch.from_numpy(X_train).cuda(),
        torch.from_numpy(alpha).cuda(),
        torch.from_numpy(L).cuda(),
        lengthscale, variance, y_best, exploration_param, acq_type
    )
    return res.cpu().numpy()
