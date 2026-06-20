"""
tests/test_cholesky_kernels.py
Correctness tests for:
  1. acquisition_fused_chol  — Cholesky-based fused acquisition (O(n) SMEM)
  2. rbf_kernel_tiled_fp16   — Mixed-precision (FP16 inputs) tiled RBF
"""

import numpy as np
import pytest
import scipy.linalg
import torch

from bayesopt_cuda.reference import rbf_kernel
from bayesopt_cuda.kernels_optimized import (
    acquisition_fused,
    acquisition_fused_chol,
    rbf_kernel_tiled,
    rbf_kernel_tiled_fp16,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_gp_state(n: int, d: int, seed: int = 42):
    """Return a consistent set of GP inputs for testing."""
    rng = np.random.default_rng(seed)
    X_train = rng.normal(size=(n, d)).astype(np.float32)
    Y_train = rng.normal(size=(n,)).astype(np.float32)
    ls, var, noise = 1.2, 1.5, 1e-4

    K = rbf_kernel(X_train, X_train, ls, var)
    K[np.diag_indices_from(K)] += noise

    L = scipy.linalg.cholesky(K, lower=True).astype(np.float32)
    alpha = scipy.linalg.cho_solve((L, True), Y_train).astype(np.float32)

    L_inv = scipy.linalg.solve_triangular(L, np.eye(n), lower=True)
    K_inv = (L_inv.T @ L_inv).astype(np.float32)

    return X_train, alpha, L, K_inv, ls, var


# ---------------------------------------------------------------------------
# 1. acquisition_fused_chol correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("acq_type", ["ei", "ucb", "pi"])
@pytest.mark.parametrize("n,m,d", [
    (10,  20,  2),
    (32,  128, 4),
    (64,  256, 4),
    (128, 512, 8),   # n=128 would SMEM-spill the old K_inv kernel
    (256, 256, 16),  # n=256 completely infeasible with old kernel
    (512, 64,  8),   # n=512 — 1 MB K_inv SMEM before, 4 KB now
])
def test_acquisition_chol_vs_kinv(n, m, d, acq_type):
    """Cholesky acquisition must match the K_inv acquisition within 1e-2."""
    rng = np.random.default_rng(7)
    X_test = rng.normal(size=(m, d)).astype(np.float32)
    X_train, alpha, L, K_inv, ls, var = make_gp_state(n, d)
    y_best = float(np.max(alpha))  # arbitrary positive reference
    xi = 0.1

    # Reference: K_inv-based fused kernel (known good for small n)
    # For n > 64 this will likely produce garbage (SMEM spill), so we
    # use the NumPy reference instead for large n.
    if n <= 64:
        ref = acquisition_fused(
            torch.from_numpy(X_test).cuda(),
            torch.from_numpy(X_train).cuda(),
            torch.from_numpy(alpha).cuda(),
            torch.from_numpy(K_inv).cuda(),
            ls, var, y_best, xi, acq_type
        ).cpu().numpy()
    else:
        # Compute reference via NumPy/SciPy directly
        K_cross = rbf_kernel(X_test, X_train, ls, var)
        mean_ref = (K_cross @ alpha).astype(np.float32)
        V = K_cross @ K_inv
        var_ref = np.clip(var - np.sum(V * K_cross, axis=1), 0, None).astype(np.float32)
        std_ref = np.sqrt(var_ref)
        from bayesopt_cuda.reference import (
            expected_improvement, upper_confidence_bound, probability_of_improvement
        )
        if acq_type == "ei":
            ref = expected_improvement(mean_ref, std_ref, y_best, xi)
        elif acq_type == "ucb":
            ref = upper_confidence_bound(mean_ref, std_ref, xi)
        else:
            ref = probability_of_improvement(mean_ref, std_ref, y_best, xi)
        ref = ref.astype(np.float32)

    # Cholesky-based kernel
    chol = acquisition_fused_chol(
        torch.from_numpy(X_test).cuda(),
        torch.from_numpy(X_train).cuda(),
        torch.from_numpy(alpha).cuda(),
        torch.from_numpy(L).cuda(),
        ls, var, y_best, xi, acq_type
    ).cpu().numpy()

    np.testing.assert_allclose(
        ref, chol, rtol=1e-2, atol=1e-2,
        err_msg=f"Cholesky acquisition mismatch vs reference for n={n},m={m},d={d},{acq_type}"
    )


def test_acquisition_chol_ei_non_negative():
    """EI from Cholesky kernel must always be >= 0."""
    n, m, d = 128, 512, 8
    rng = np.random.default_rng(99)
    X_test = rng.normal(size=(m, d)).astype(np.float32)
    X_train, alpha, L, _, ls, var = make_gp_state(n, d)
    y_best = float(np.percentile(alpha, 50))

    ei = acquisition_fused_chol(
        torch.from_numpy(X_test).cuda(),
        torch.from_numpy(X_train).cuda(),
        torch.from_numpy(alpha).cuda(),
        torch.from_numpy(L).cuda(),
        ls, var, y_best, 0.01, "ei"
    ).cpu().numpy()
    assert np.all(ei >= 0), f"Negative EI values: {ei[ei < 0]}"


def test_acquisition_chol_pi_in_01():
    """PI from Cholesky kernel must be in [0, 1]."""
    n, m, d = 64, 256, 4
    rng = np.random.default_rng(11)
    X_test = rng.normal(size=(m, d)).astype(np.float32)
    X_train, alpha, L, _, ls, var = make_gp_state(n, d)

    pi = acquisition_fused_chol(
        torch.from_numpy(X_test).cuda(),
        torch.from_numpy(X_train).cuda(),
        torch.from_numpy(alpha).cuda(),
        torch.from_numpy(L).cuda(),
        ls, var, 0.0, 0.01, "pi"
    ).cpu().numpy()
    assert np.all(pi >= 0) and np.all(pi <= 1), "PI must be in [0, 1]"


# ---------------------------------------------------------------------------
# 2. rbf_kernel_tiled_fp16 correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,m,d", [
    (64,   64,  4),
    (256,  512, 8),
    (1024, 1024, 16),
    (2048, 2048, 32),
])
def test_rbf_fp16_vs_fp32(n, m, d):
    """FP16 tiled RBF must match FP32 tiled RBF within 1% relative error."""
    rng = np.random.default_rng(17)
    # Use bounded values to avoid FP16 overflow (|diff| < 256 for safety)
    X32 = (rng.normal(size=(n, d)) * 2).astype(np.float32)
    Y32 = (rng.normal(size=(m, d)) * 2).astype(np.float32)
    ls, var = 1.5, 2.0

    X_t32 = torch.from_numpy(X32).cuda()
    Y_t32 = torch.from_numpy(Y32).cuda()
    X_t16 = X_t32.half()
    Y_t16 = Y_t32.half()

    K_fp32 = rbf_kernel_tiled(X_t32, Y_t32, ls, var).cpu().numpy()
    K_fp16 = rbf_kernel_tiled_fp16(X_t16, Y_t16, ls, var).cpu().numpy()

    np.testing.assert_allclose(
        K_fp32, K_fp16, rtol=2e-2, atol=1e-3,
        err_msg=f"FP16 RBF mismatch vs FP32 at n={n},m={m},d={d}"
    )


def test_rbf_fp16_output_dtype():
    """FP16 tiled RBF must return float32 output."""
    X = torch.randn(32, 4, dtype=torch.float16, device="cuda")
    Y = torch.randn(32, 4, dtype=torch.float16, device="cuda")
    K = rbf_kernel_tiled_fp16(X, Y, 1.0, 1.0)
    assert K.dtype == torch.float32, f"Expected float32 output, got {K.dtype}"


def test_rbf_fp16_symmetry():
    """K(X, X) must be symmetric for FP16 kernel."""
    n, d = 128, 8
    X = torch.randn(n, d, dtype=torch.float16, device="cuda")
    K = rbf_kernel_tiled_fp16(X, X, 1.0, 1.0)
    np.testing.assert_allclose(
        K.cpu().numpy(), K.T.cpu().numpy(), rtol=1e-3, atol=1e-3,
        err_msg="FP16 RBF kernel is not symmetric"
    )
