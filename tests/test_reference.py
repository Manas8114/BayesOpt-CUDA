"""Stage 1 reference tests."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from bayesopt_cuda.reference import (
    rbf_kernel,
    GaussianProcessRegressor,
    expected_improvement,
    upper_confidence_bound,
    probability_of_improvement,
    branin,
    hartmann6,
    BayesianOptimizer,
)


def test_rbf_identical_points():
    X = np.array([[1.0, 2.0]])
    K = rbf_kernel(X, X, lengthscale=1.0, variance=2.5)
    assert np.allclose(K[0, 0], 2.5)


def test_rbf_symmetry():
    X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    Y = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=float)
    K1 = rbf_kernel(X, Y, lengthscale=2.0, variance=1.0)
    K2 = rbf_kernel(Y, X, lengthscale=2.0, variance=1.0)
    assert np.allclose(K1, K2.T)


def test_rbf_decay():
    X = np.array([[0.0, 0.0]])
    Y1 = np.array([[1.0, 0.0]])
    Y2 = np.array([[2.0, 0.0]])
    K1 = rbf_kernel(X, Y1, lengthscale=1.0, variance=1.0)
    K2 = rbf_kernel(X, Y2, lengthscale=1.0, variance=1.0)
    assert K1[0, 0] > K2[0, 0]


def test_rbf_lengthscale_sensitivity():
    X = np.array([[0.0, 0.0]])
    Y = np.array([[2.0, 0.0]])
    Ks = rbf_kernel(X, Y, lengthscale=1.0, variance=1.0)[0, 0]
    Kl = rbf_kernel(X, Y, lengthscale=2.0, variance=1.0)[0, 0]
    assert Kl > Ks


def test_rbf_variance_scaling():
    X = np.array([[0.0, 0.0]])
    Y = np.array([[0.5, 0.0]])
    K1 = rbf_kernel(X, Y, lengthscale=1.0, variance=1.0)
    K2 = rbf_kernel(X, Y, lengthscale=1.0, variance=2.5)
    assert np.allclose(K2, 2.5 * K1)


def test_rbf_random_match_naive():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(33, 5)).astype(np.float64)
    Y = rng.normal(size=(44, 5)).astype(np.float64)
    lengthscale, variance = 1.3, 2.1
    K = rbf_kernel(X, Y, lengthscale, variance)
    expected = np.zeros((33, 44))
    for i in range(33):
        for j in range(44):
            sq = np.sum((X[i] - Y[j]) ** 2)
            expected[i, j] = variance * np.exp(-0.5 * sq / lengthscale ** 2)
    assert np.allclose(K, expected)


def test_gp_interpolation():
    X = np.array([[0.5], [1.5]], dtype=float)
    y = np.array([1.7, -0.3])
    gp = GaussianProcessRegressor(lengthscale=1.0, variance=1.0, noise=1e-4)
    gp.fit(X, y)
    mean, _std = gp.predict(X)
    assert np.allclose(mean, y, atol=1e-3)


def test_gp_log_marginal_likelihood():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 2))
    y = np.sin(X[:, 0]) + 0.1 * rng.normal(size=10)
    gp = GaussianProcessRegressor(lengthscale=1.0, variance=1.0, noise=1e-4)
    gp.fit(X, y)
    assert np.isfinite(gp.log_marginal_likelihood())


def test_ei_hand_validated():
    from scipy.stats import norm
    mean = np.array([1.5])
    std = np.array([0.4])
    ei = expected_improvement(mean, std, y_best=1.0, xi=0.1)
    expected = 0.4 * norm.cdf(1.0) + 0.4 * norm.pdf(1.0)
    assert np.allclose(ei[0], expected, atol=1e-6)


def test_ei_std_zero_returns_zero():
    mean = np.array([0.5, 1.0, 1.5])
    std = np.array([0.0, 0.0, 0.0])
    ei = expected_improvement(mean, std, y_best=1.0, xi=0.01)
    assert np.allclose(ei, 0.0)


def test_ucb_basic():
    mean = np.array([0.5, 1.0])
    std = np.array([0.2, 0.4])
    beta = 2.0
    ucb = upper_confidence_bound(mean, std, beta=beta)
    assert np.allclose(ucb, mean + np.sqrt(beta) * std)


def test_pi_basic():
    mean = np.array([0.5, 1.0, 1.5])
    std = np.array([0.2, 0.1, 0.3])
    pi = probability_of_improvement(mean, std, y_best=1.0, xi=0.01)
    assert np.all((pi >= 0) & (pi <= 1))


def test_branin_min_value():
    pts = [np.array([-np.pi, 12.275]),
           np.array([np.pi, 2.275]),
           np.array([9.42478, 2.475])]
    for p in pts:
        assert abs(branin(p) - 0.397887) < 1e-3, f"branin({p})"


def test_hartmann6_min():
    x = np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573])
    assert abs(hartmann6(x) - (-3.32237)) < 1e-3


# Stage 1 verification gate.
# The Branin function has 3 separate global minima (all at value 0.397887).
# With the reference BO hyperparameters we converge to within 0.1 of a
# global minimum after ~25 iterations on this 2-D test function. The
# tolerance is generous to accommodate the stochastic L-BFGS-B restart
# behaviour; the tighter the n_iterations, the tighter the test.
def test_branin_bo_converges():
    """End-to-end BO finds Branin minimum within 0.15 absolute error in 25 iters."""
    bounds = np.array([[-5.0, 10.0], [0.0, 15.0]])
    opt = BayesianOptimizer(
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
    opt.run(25)
    best_val = -opt.best_y
    assert abs(best_val - 0.397887) < 0.15, (
        f"BO failed to converge to Branin global min; "
        f"best={best_val:.4f}, target=0.397887"
    )
