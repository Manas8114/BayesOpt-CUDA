import sys
from pathlib import Path
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from bayesopt_cuda.reference import (
    rbf_kernel,
    expected_improvement,
    upper_confidence_bound,
    probability_of_improvement,
)
from bayesopt_cuda.kernels_naive import (
    rbf_kernel_naive_np,
    expected_improvement_naive_np,
    upper_confidence_bound_naive_np,
    probability_of_improvement_naive_np,
)

if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)


@pytest.mark.parametrize("n, m, d", [
    (10, 10, 2),     # tiny
    (100, 50, 5),    # medium
    (1000, 200, 10)  # large
])
def test_rbf_kernel_sizes(n, m, d):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n, d)).astype(np.float32)
    Y = rng.normal(size=(m, d)).astype(np.float32)
    
    lengthscale = 1.5
    variance = 2.0
    
    K_ref = rbf_kernel(X, Y, lengthscale, variance)
    K_naive = rbf_kernel_naive_np(X, Y, lengthscale, variance)
    
    assert K_naive.shape == (n, m)
    assert K_naive.dtype == np.float32
    assert not np.any(np.isnan(K_naive))
    assert not np.any(np.isinf(K_naive))
    assert np.allclose(K_naive, K_ref, rtol=1e-4, atol=1e-5)


def test_rbf_kernel_identical_points():
    X = np.array([[1.0, 2.0]], dtype=np.float32)
    lengthscale = 1.0
    variance = 2.5
    
    K_ref = rbf_kernel(X, X, lengthscale, variance)
    K_naive = rbf_kernel_naive_np(X, X, lengthscale, variance)
    
    assert np.allclose(K_naive, K_ref, rtol=1e-4, atol=1e-5)
    assert np.allclose(K_naive[0, 0], 2.5)


def test_rbf_kernel_single_training_point():
    X = np.array([[1.0, 2.0]], dtype=np.float32)
    Y = np.array([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    
    K_ref = rbf_kernel(X, Y)
    K_naive = rbf_kernel_naive_np(X, Y)
    
    assert K_naive.shape == (1, 2)
    assert np.allclose(K_naive, K_ref, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("m", [
    10,      # tiny
    500,     # medium
    10000    # large
])
def test_ei_sizes(m):
    rng = np.random.default_rng(42)
    mean = rng.normal(size=(m,)).astype(np.float32)
    std = rng.uniform(0.1, 2.0, size=(m,)).astype(np.float32)
    y_best = 1.0
    xi = 0.01
    
    ei_ref = expected_improvement(mean, std, y_best, xi)
    ei_naive = expected_improvement_naive_np(mean, std, y_best, xi)
    
    assert ei_naive.shape == (m,)
    assert ei_naive.dtype == np.float32
    assert not np.any(np.isnan(ei_naive))
    assert not np.any(np.isinf(ei_naive))
    assert np.allclose(ei_naive, ei_ref, rtol=1e-4, atol=1e-5)


def test_ei_std_zero():
    mean = np.array([0.5, 1.0, 1.5], dtype=np.float32)
    std = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    y_best = 1.0
    xi = 0.01
    
    ei_ref = expected_improvement(mean, std, y_best, xi)
    ei_naive = expected_improvement_naive_np(mean, std, y_best, xi)
    
    assert np.allclose(ei_naive, ei_ref, rtol=1e-4, atol=1e-5)
    assert np.allclose(ei_naive, 0.0)


@pytest.mark.parametrize("m", [10, 500, 10000])
def test_ucb_sizes(m):
    rng = np.random.default_rng(42)
    mean = rng.normal(size=(m,)).astype(np.float32)
    std = rng.uniform(0.1, 2.0, size=(m,)).astype(np.float32)
    beta = 2.0
    
    ucb_ref = upper_confidence_bound(mean, std, beta)
    ucb_naive = upper_confidence_bound_naive_np(mean, std, beta)
    
    assert ucb_naive.shape == (m,)
    assert np.allclose(ucb_naive, ucb_ref, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("m", [10, 500, 10000])
def test_pi_sizes(m):
    rng = np.random.default_rng(42)
    mean = rng.normal(size=(m,)).astype(np.float32)
    std = rng.uniform(0.1, 2.0, size=(m,)).astype(np.float32)
    y_best = 1.0
    xi = 0.01
    
    pi_ref = probability_of_improvement(mean, std, y_best, xi)
    pi_naive = probability_of_improvement_naive_np(mean, std, y_best, xi)
    
    assert pi_naive.shape == (m,)
    assert np.allclose(pi_naive, pi_ref, rtol=1e-4, atol=1e-5)
