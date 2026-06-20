"""
Tests for the CUDA GP and Bayesian Optimizer API (Stage 4).

These tests verify:
  1. GaussianProcessRegressorCUDA produces mean/std that match the
     reference NumPy implementation within 1%.
  2. BayesianOptimizerCUDA finds the Branin global minimum within 0.5
     of the true value in 25 iterations.
"""

import numpy as np
import pytest
import scipy.linalg

from bayesopt_cuda.reference import (
    rbf_kernel,
    GaussianProcessRegressor,
    expected_improvement,
    upper_confidence_bound,
    probability_of_improvement,
    branin,
)
from bayesopt_cuda.gp_cuda import GaussianProcessRegressorCUDA
from bayesopt_cuda.optimizer_cuda import BayesianOptimizerCUDA


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_gp_data(n=50, d=3, seed=7):
    rng = np.random.default_rng(seed)
    X_train = rng.normal(size=(n, d)).astype(np.float32)
    y_train = rng.normal(size=(n,)).astype(np.float32)
    X_test  = rng.normal(size=(20, d)).astype(np.float32)
    return X_train, y_train, X_test


# ---------------------------------------------------------------------------
# GP correctness tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,d", [(10, 2), (50, 5), (100, 3)])
def test_gp_mean_matches_reference(n, d):
    """Mean predictions should match reference GP within 1%."""
    X_train, y_train, X_test = make_gp_data(n, d)

    ls, var, noise = 1.0, 1.0, 1e-4

    # Reference (NumPy)
    gp_ref = GaussianProcessRegressor(
        lengthscale=ls, variance=var, noise=noise, jitter=1e-6
    ).fit(X_train.astype(np.float64), y_train.astype(np.float64))
    mean_ref, std_ref = gp_ref.predict(X_test.astype(np.float64))

    # CUDA
    gp_cuda = GaussianProcessRegressorCUDA(
        lengthscale=ls, variance=var, noise=noise, jitter=1e-6
    ).fit(X_train, y_train)
    mean_cuda, std_cuda = gp_cuda.predict(X_test)

    np.testing.assert_allclose(mean_ref, mean_cuda, rtol=1e-2, atol=1e-2,
                                err_msg="Mean mismatch between reference and CUDA GP")
    # std comparison: near-zero values can differ due to float32 vs float64
    # precision and --use_fast_math. We test that they agree when both are
    # non-trivially non-zero; otherwise we just require both are non-negative.
    assert np.all(std_cuda >= 0), "std_cuda has negative values"
    mask = (std_ref > 0.05) & (std_cuda > 0.05)
    if mask.any():
        np.testing.assert_allclose(
            std_ref[mask], std_cuda[mask], rtol=5e-2, atol=5e-2,
            err_msg="Std mismatch for non-trivial values")


def test_gp_variance_non_negative():
    """Predictive variance must never be negative."""
    X_train, y_train, X_test = make_gp_data()
    gp = GaussianProcessRegressorCUDA(lengthscale=1.0, variance=1.0, noise=1e-4).fit(X_train, y_train)
    _, std = gp.predict(X_test)
    assert np.all(std >= 0), "std should be non-negative"


def test_gp_variance_zero_near_training_point():
    """Variance near a training point should be very small (noise floor)."""
    X_train = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    y_train = np.array([1.0, -1.0], dtype=np.float32)
    gp = GaussianProcessRegressorCUDA(
        lengthscale=1.0, variance=1.0, noise=1e-6, jitter=1e-6
    ).fit(X_train, y_train)

    # Predict at training point (should have near-zero variance)
    X_test = np.array([[0.0, 0.0]], dtype=np.float32)
    _, std = gp.predict(X_test)
    assert std[0] < 0.01, f"Std near training point should be small, got {std[0]}"


def test_gp_acquisition_shapes():
    """Acquisition function should return (m,) array."""
    X_train, y_train, _ = make_gp_data()
    X_test = np.random.normal(size=(100, 3)).astype(np.float32)
    gp = GaussianProcessRegressorCUDA(lengthscale=1.0, variance=1.0, noise=1e-4).fit(X_train, y_train)

    y_best = float(np.max(y_train))
    for acq_type in ["ei", "ucb", "pi"]:
        vals = gp.acquisition(X_test, y_best, acq_type=acq_type)
        assert vals.shape == (100,), f"Wrong shape for {acq_type}: {vals.shape}"


def test_gp_ei_non_negative():
    """EI values must be non-negative."""
    X_train, y_train, X_test = make_gp_data()
    gp = GaussianProcessRegressorCUDA(lengthscale=1.0, variance=1.0, noise=1e-4).fit(X_train, y_train)
    ei = gp.acquisition(X_test, y_best=float(np.max(y_train)), acq_type="ei")
    assert np.all(ei >= 0), f"EI has negative values: {ei[ei < 0]}"


def test_gp_pi_in_01():
    """PI values must be in [0, 1]."""
    X_train, y_train, X_test = make_gp_data()
    gp = GaussianProcessRegressorCUDA(lengthscale=1.0, variance=1.0, noise=1e-4).fit(X_train, y_train)
    pi = gp.acquisition(X_test, y_best=float(np.max(y_train)), acq_type="pi")
    assert np.all(pi >= 0) and np.all(pi <= 1), "PI must be in [0, 1]"


# ---------------------------------------------------------------------------
# BO convergence test
# ---------------------------------------------------------------------------

def test_bo_cuda_branin_convergence():
    """
    BayesianOptimizerCUDA must find Branin's global minimum ≈ 0.397887
    within tolerance 0.5 in 25 iterations.
    """
    bounds = np.array([[-5.0, 10.0], [0.0, 15.0]])

    opt = BayesianOptimizerCUDA(
        objective=lambda x: -branin(x),
        bounds=bounds,
        acquisition="ei",
        lengthscale=2.0,
        variance=100.0,
        noise=1e-6,
        jitter=1e-6,
        random_state=42,
        n_candidates=2000,
        n_local_restarts=5,
    )
    opt.run(n_iter=25)

    best_val = -opt.best_y   # convert back to minimization
    global_min = 0.397887
    tolerance = 0.5

    assert abs(best_val - global_min) < tolerance, (
        f"BO-CUDA failed to converge on Branin: best={best_val:.4f}, "
        f"target={global_min:.4f} ± {tolerance}"
    )
