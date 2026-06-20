import pytest
import numpy as np
import torch
import scipy.linalg

from bayesopt_cuda.reference import (
    rbf_kernel,
    expected_improvement,
    upper_confidence_bound,
    probability_of_improvement,
)
from bayesopt_cuda.kernels_optimized import (
    rbf_kernel_tiled_np,
    acquisition_fused_np,
)

# Test RBF kernel tiled vs NumPy reference
@pytest.mark.parametrize("n,m,d", [
    (10, 10, 2),
    (100, 50, 5),
    (1000, 200, 10),
])
def test_rbf_kernel_tiled(n, m, d):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n, d)).astype(np.float32)
    Y = rng.normal(size=(m, d)).astype(np.float32)
    lengthscale = 1.5
    variance = 2.0

    K_ref = rbf_kernel(X, Y, lengthscale, variance)
    K_tiled = rbf_kernel_tiled_np(X, Y, lengthscale, variance)

    np.testing.assert_allclose(K_ref, K_tiled, rtol=1e-4, atol=1e-4)

# Test fused acquisition vs NumPy reference
@pytest.mark.parametrize("acq_type", ["ei", "ucb", "pi"])
@pytest.mark.parametrize("n,m,d", [
    (10, 20, 2),
    (100, 500, 5),
])
def test_acquisition_fused(acq_type, n, m, d):
    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(n, d)).astype(np.float32)
    X_test = rng.normal(size=(m, d)).astype(np.float32)
    
    Y_train = rng.normal(size=(n,)).astype(np.float32)
    
    lengthscale = 1.2
    variance = 1.5
    y_best = float(np.max(Y_train))
    exploration_param = 0.1

    # 1. Compute reference mean and std using NumPy
    # Add small noise to diagonal for stability
    K_train = rbf_kernel(X_train, X_train, lengthscale, variance)
    K_train[np.diag_indices_from(K_train)] += 1e-6
    
    L = scipy.linalg.cholesky(K_train, lower=True)
    alpha = scipy.linalg.cho_solve((L, True), Y_train).astype(np.float32)
    
    # Compute K_inv = L^{-T} L^{-1}
    L_inv = scipy.linalg.solve_triangular(L, np.eye(n), lower=True)
    K_inv = (L_inv.T @ L_inv).astype(np.float32)

    K_cross = rbf_kernel(X_test, X_train, lengthscale, variance)
    
    mean_ref = K_cross @ alpha
    var_ref = variance - np.sum((K_cross @ K_inv) * K_cross, axis=1)
    var_ref = np.clip(var_ref, 0, None)
    std_ref = np.sqrt(var_ref).astype(np.float32)

    if acq_type == "ei":
        acq_ref = expected_improvement(mean_ref, std_ref, y_best, exploration_param)
    elif acq_type == "ucb":
        acq_ref = upper_confidence_bound(mean_ref, std_ref, exploration_param)
    elif acq_type == "pi":
        acq_ref = probability_of_improvement(mean_ref, std_ref, y_best, exploration_param)

    # 2. Compute using fused kernel
    acq_fused = acquisition_fused_np(
        X_test, X_train, alpha, K_inv,
        lengthscale, variance, y_best, exploration_param, acq_type
    )

    # Allow some leniency due to Fast Math exp/sqrt differences.
    # For EI with near-zero std, both implementations should agree on max(0, delta).
    np.testing.assert_allclose(acq_ref, acq_fused, rtol=1e-2, atol=1e-2)
