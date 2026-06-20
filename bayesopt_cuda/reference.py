"""
Pure Python/NumPy/SciPy reference implementation of Bayesian Optimization.

This is the ground truth for all CUDA kernel correctness checks.
Hard rules:
- No CUDA, no PyTorch GPU, no compiled extensions
- NumPy/SciPy only
- Must find Branin global minimum within documented tolerance
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from typing import Callable, Optional, Tuple, List
import warnings


# ============================================================================
# Kernel Functions
# ============================================================================

def rbf_kernel(X: np.ndarray, Y: np.ndarray, lengthscale: float = 1.0, variance: float = 1.0) -> np.ndarray:
    """
    RBF (Squared Exponential) kernel between two point sets.
    
    k(x, y) = variance * exp(-||x - y||^2 / (2 * lengthscale^2))
    
    Args:
        X: (n, d) array
        Y: (m, d) array
        lengthscale: kernel lengthscale
        variance: kernel variance (signal variance)
    
    Returns:
        (n, m) kernel matrix
    """
    X_norm = np.sum(X ** 2, axis=1, keepdims=True)  # (n, 1)
    Y_norm = np.sum(Y ** 2, axis=1, keepdims=True).T  # (1, m)
    XY = X @ Y.T  # (n, m)
    sqdist = X_norm + Y_norm - 2 * XY  # (n, m)
    sqdist = np.maximum(sqdist, 0.0)
    
    return variance * np.exp(-0.5 * sqdist / (lengthscale ** 2))


# ============================================================================
# Gaussian Process Regression (Cholesky-based)
# ============================================================================

class GaussianProcessRegressor:
    """
    GP regression with RBF kernel, Cholesky decomposition, and explicit jitter.
    
    This is the reference implementation that CUDA kernels must match numerically.
    """
    
    def __init__(
        self,
        lengthscale: float = 1.0,
        variance: float = 1.0,
        noise: float = 1e-6,
        jitter: float = 1e-6,
    ):
        self.lengthscale = lengthscale
        self.variance = variance
        self.noise = noise  # observation noise (added to diagonal)
        self.jitter = jitter  # numerical jitter for Cholesky stability
        
        self.X_train_: Optional[np.ndarray] = None
        self.y_train_: Optional[np.ndarray] = None
        self.L_: Optional[np.ndarray] = None  # Cholesky factor
        self.alpha_: Optional[np.ndarray] = None  # K^{-1} y
        self.is_fitted_ = False
    
    def _kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return rbf_kernel(X, Y, self.lengthscale, self.variance)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianProcessRegressor":
        """
        Fit GP to training data.
        
        Args:
            X: (n, d) training inputs
            y: (n,) training targets
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have same length: {X.shape[0]} vs {y.shape[0]}")
        
        n = X.shape[0]
        self.X_train_ = X
        self.y_train_ = y
        
        # Compute kernel matrix
        K = self._kernel(X, X)
        
        # Add noise + jitter to diagonal for numerical stability
        # noise: observation noise (part of model)
        # jitter: extra regularization for Cholesky
        K.flat[::n + 1] += self.noise + self.jitter
        
        # Cholesky decomposition: K = L L^T
        try:
            self.L_ = np.linalg.cholesky(K)
        except np.linalg.LinAlgError as e:
            # If Cholesky fails, increase jitter and retry once
            warnings.warn(f"Cholesky failed ({e}), increasing jitter and retrying")
            K.flat[::n + 1] += 1e-4
            self.L_ = np.linalg.cholesky(K)
        
        # Solve L L^T alpha = y  ->  L^T alpha = L^{-1} y
        # alpha = K^{-1} y
        self.alpha_ = np.linalg.solve(self.L_.T, np.linalg.solve(self.L_, y))
        
        self.is_fitted_ = True
        return self
    
    def predict(
        self,
        X: np.ndarray,
        return_std: bool = True,
        return_cov: bool = False,
    ) -> Tuple[np.ndarray, ...]:
        """
        Predict at test points.
        
        Args:
            X: (m, d) test inputs
            return_std: if True, return predictive standard deviation
            return_cov: if True, return full predictive covariance
            
        Returns:
            mean: (m,) predictive mean
            std: (m,) predictive standard deviation (if return_std)
            cov: (m, m) predictive covariance (if return_cov)
        """
        if not self.is_fitted_:
            raise RuntimeError("GP must be fitted before prediction")
        
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # K_trans = k(X_test, X_train)  (m, n)
        K_trans = self._kernel(X, self.X_train_)
        
        # Mean: K_trans @ alpha
        mean = K_trans @ self.alpha_
        
        if not return_std and not return_cov:
            return mean
        
        # Variance: k(X_test, X_test) - K_trans @ K^{-1} @ K_trans.T
        # Solve L v = K_trans.T  ->  v = L^{-1} K_trans.T
        v = np.linalg.solve(self.L_, K_trans.T)
        
        # Predictive covariance: K_test - v.T @ v
        K_test = self._kernel(X, X)
        cov = K_test - v.T @ v
        
        # Numerical stability: clip negative variances
        var = np.diag(cov)
        var = np.maximum(var, 0.0)
        std = np.sqrt(var)
        
        if return_cov:
            return mean, std, cov
        return mean, std
    
    def log_marginal_likelihood(self) -> float:
        """Compute log marginal likelihood of training data."""
        if not self.is_fitted_:
            raise RuntimeError("GP must be fitted first")
        n = self.X_train_.shape[0]
        # log p(y|X) = -0.5 * y^T K^{-1} y - sum(log(diag(L))) - n/2 * log(2*pi)
        # alpha = K^{-1} y, so y^T alpha = y^T K^{-1} y
        yTK_inv_y = self.y_train_ @ self.alpha_
        logdet = 2 * np.sum(np.log(np.diag(self.L_)))
        return -0.5 * yTK_inv_y - logdet - 0.5 * n * np.log(2 * np.pi)


# ============================================================================
# Acquisition Functions
# ============================================================================

def expected_improvement(
    mean: np.ndarray,
    std: np.ndarray,
    y_best: float,
    xi: float = 0.01,
) -> np.ndarray:
    """
    Expected Improvement (EI) acquisition function.
    
    EI(x) = E[max(0, f(x) - y_best - xi)]
          = (mean - y_best - xi) * Phi(Z) + std * phi(Z)
          where Z = (mean - y_best - xi) / std
    
    Args:
        mean: (m,) predictive mean
        std: (m,) predictive standard deviation
        y_best: best observed value so far (maximum for maximization)
        xi: exploration parameter
    
    Returns:
        (m,) EI values
    """
    # Handle std = 0 case (exact interpolation at training points)
    std = np.maximum(std, 1e-12)
    
    improvement = mean - y_best - xi
    Z = improvement / std
    
    # Phi = CDF, phi = PDF
    Phi = norm.cdf(Z)
    phi = norm.pdf(Z)
    
    ei = improvement * Phi + std * phi
    
    # Where std == 0, EI = 0 (no improvement possible if we already know the value)
    ei = np.where(std > 1e-12, ei, 0.0)
    
    # For minimization, flip sign; here we assume maximization
    return ei


def upper_confidence_bound(
    mean: np.ndarray,
    std: np.ndarray,
    beta: float = 2.0,
) -> np.ndarray:
    """
    Upper Confidence Bound (UCB) acquisition function.
    
    UCB(x) = mean + sqrt(beta) * std
    
    Args:
        mean: (m,) predictive mean
        std: (m,) predictive standard deviation
        beta: exploration-exploitation tradeoff parameter
    
    Returns:
        (m,) UCB values
    """
    return mean + np.sqrt(beta) * std


def probability_of_improvement(
    mean: np.ndarray,
    std: np.ndarray,
    y_best: float,
    xi: float = 0.01,
) -> np.ndarray:
    """
    Probability of Improvement (PI) acquisition function.
    
    PI(x) = P(f(x) > y_best + xi) = Phi((mean - y_best - xi) / std)
    
    Args:
        mean: (m,) predictive mean
        std: (m,) predictive standard deviation
        y_best: best observed value so far
        xi: exploration parameter
    
    Returns:
        (m,) PI values
    """
    std = np.maximum(std, 1e-12)
    Z = (mean - y_best - xi) / std
    return norm.cdf(Z)


# ============================================================================
# Test Functions (Branin, Hartmann6)
# ============================================================================

def branin(x: np.ndarray) -> float:
    """
    Branin function (2D) - standard test function for BO.
    
    Global minima: f(x1, x2) = 0.397887 at:
    - (-pi, 12.275), (pi, 2.275), (9.42478, 2.475)
    Domain: x1 in [-5, 10], x2 in [0, 15]
    """
    x = np.asarray(x).ravel()
    if x.shape[0] != 2:
        raise ValueError("Branin requires 2D input")
    x1, x2 = x[0], x[1]
    
    a = 1.0
    b = 5.1 / (4 * np.pi ** 2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8 * np.pi)
    
    term1 = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2
    term2 = s * (1 - t) * np.cos(x1)
    term3 = s
    
    return term1 + term2 + term3


def hartmann6(x: np.ndarray) -> float:
    """
    Hartmann6 function (6D) - standard test function for BO.
    
    Global minimum: f(x) = -3.32237 at x* = (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)
    Domain: x in [0, 1]^6
    """
    x = np.asarray(x).ravel()
    if x.shape[0] != 6:
        raise ValueError("Hartmann6 requires 6D input")
    
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([
        [10, 3, 17, 3.5, 1.7, 8],
        [0.05, 10, 17, 0.1, 8, 14],
        [3, 3.5, 1.7, 10, 17, 8],
        [17, 8, 0.05, 10, 0.1, 14],
    ])
    P = 1e-4 * np.array([
        [1312, 1696, 5569, 124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381],
    ])
    
    outer = np.sum(A * (x - P) ** 2, axis=1)
    return -np.sum(alpha * np.exp(-outer))


# ============================================================================
# Bayesian Optimization Loop
# ============================================================================

class BayesianOptimizer:
    """
    Pure Python Bayesian Optimization loop.
    
    This is the reference implementation that the CUDA-accelerated version
    must match in behavior (not necessarily speed).
    """
    
    def __init__(
        self,
        objective: Callable[[np.ndarray], float],
        bounds: np.ndarray,
        acquisition: str = "ei",
        lengthscale: float = 1.0,
        variance: float = 1.0,
        noise: float = 1e-6,
        jitter: float = 1e-6,
        xi: float = 0.01,
        beta: float = 2.0,
        random_state: Optional[int] = None,
        n_candidates: int = 5000,
        n_local_restarts: int = 10,
    ):
        """
        Args:
            objective: function to minimize (takes 1D array, returns float)
            bounds: (d, 2) array of [min, max] for each dimension
            acquisition: "ei", "ucb", or "pi"
            lengthscale: RBF kernel lengthscale
            variance: RBF kernel variance
            noise: observation noise
            jitter: numerical jitter for Cholesky
            xi: exploration parameter for EI/PI
            beta: exploration parameter for UCB
            random_state: seed for reproducibility
            n_candidates: number of random candidates for acquisition optimization
            n_local_restarts: number of top candidates to refine with L-BFGS-B
        """
        self.objective = objective
        self.bounds = np.asarray(bounds, dtype=np.float64)
        self.acquisition_name = acquisition
        self.lengthscale = lengthscale
        self.variance = variance
        self.noise = noise
        self.jitter = jitter
        self.xi = xi
        self.beta = beta
        self.n_candidates = n_candidates
        self.n_local_restarts = n_local_restarts
        
        if random_state is not None:
            np.random.seed(random_state)
        
        self.X_: List[np.ndarray] = []
        self.y_: List[float] = []
        self.gp_: Optional[GaussianProcessRegressor] = None
        self.n_iter_ = 0
    
    def _acquisition_function(self, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """Dispatch to the selected acquisition function."""
        if self.acquisition_name == "ei":
            return expected_improvement(mean, std, np.max(self.y_), self.xi)
        elif self.acquisition_name == "ucb":
            return upper_confidence_bound(mean, std, self.beta)
        elif self.acquisition_name == "pi":
            return probability_of_improvement(mean, std, np.max(self.y_), self.xi)
        else:
            raise ValueError(f"Unknown acquisition: {self.acquisition_name}")
    
    def _optimize_acquisition(self) -> np.ndarray:
        """
        Optimize acquisition function by random sampling + local refinement.
        """
        d = self.bounds.shape[0]
        
        # Random sampling
        X_cand = np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1], size=(self.n_candidates, d)
        )
        
        # Evaluate acquisition
        mean, std = self.gp_.predict(X_cand, return_std=True)
        acq_vals = self._acquisition_function(mean, std)
        
        # Take top candidates for local optimization
        n_top = min(self.n_local_restarts, self.n_candidates)
        top_idx = np.argsort(acq_vals)[-n_top:][::-1]
        X_top = X_cand[top_idx]
        
        # Local optimization with scipy
        best_x = X_top[0]
        best_acq = acq_vals[top_idx[0]]
        
        for x0 in X_top:
            def neg_acq(x):
                x = np.asarray(x).reshape(1, -1)
                m, s = self.gp_.predict(x, return_std=True)
                return -self._acquisition_function(m, s)[0]
            
            res = minimize(
                neg_acq, x0, bounds=self.bounds.tolist(),
                method="L-BFGS-B", options={"maxiter": 100}
            )
            if res.success and -res.fun > best_acq:
                best_acq = -res.fun
                best_x = res.x
        
        return best_x
    
    def suggest(self, n_suggestions: int = 1) -> np.ndarray:
        """Suggest next point(s) to evaluate."""
        if self.gp_ is None:
            # First suggestion: random
            d = self.bounds.shape[0]
            return np.random.uniform(
                self.bounds[:, 0], self.bounds[:, 1], size=(n_suggestions, d)
            )
        
        suggestions = []
        for _ in range(n_suggestions):
            x_next = self._optimize_acquisition()
            suggestions.append(x_next)
            # Fantasize: add to GP temporarily for next suggestion
            mean, std = self.gp_.predict(x_next.reshape(1, -1), return_std=True)
            y_fantasy = mean[0]  # Use predictive mean as fantasy
            # Temporarily update GP
            X_temp = np.vstack([self.gp_.X_train_, x_next])
            y_temp = np.append(self.gp_.y_train_, y_fantasy)
            self.gp_.fit(X_temp, y_temp)
        
        # Restore original GP state
        self.gp_.fit(np.array(self.X_), np.array(self.y_))
        
        return np.array(suggestions)
    
    def step(self) -> Tuple[np.ndarray, float]:
        """Perform one BO iteration: suggest, evaluate, update."""
        x_next = self.suggest(1)[0]
        y_next = self.objective(x_next)
        
        self.X_.append(x_next)
        self.y_.append(y_next)
        
        # Refit GP
        self.gp_ = GaussianProcessRegressor(
            lengthscale=self.lengthscale,
            variance=self.variance,
            noise=self.noise,
            jitter=self.jitter,
        ).fit(np.array(self.X_), np.array(self.y_))
        
        self.n_iter_ += 1
        return x_next, y_next
    
    def run(self, n_iter: int) -> Tuple[np.ndarray, np.ndarray]:
        """Run BO for n_iter iterations."""
        for _ in range(n_iter):
            self.step()
        return np.array(self.X_), np.array(self.y_)
    
    @property
    def best_x(self) -> np.ndarray:
        """Best point found so far."""
        if not self.y_:
            raise ValueError("No evaluations yet")
        return self.X_[np.argmax(self.y_)]
    
    @property
    def best_y(self) -> float:
        """Best value found so far."""
        if not self.y_:
            raise ValueError("No evaluations yet")
        return np.max(self.y_)


# ============================================================================
# Verification / Sanity Check
# ============================================================================

def test_branin_convergence():
    """Test that BO finds Branin global minimum within tolerance."""
    bounds = np.array([[-5.0, 10.0], [0.0, 15.0]])
    
    # Branin is a minimization problem; we maximize -branin
    # Key: Branin needs larger lengthscale (~2.0) and higher variance (~100)
    # because the function varies slowly across the domain
    opt = BayesianOptimizer(
        objective=lambda x: -branin(x),
        bounds=bounds,
        acquisition="ei",
        lengthscale=2.0,  # Larger lengthscale for slow-varying Branin
        variance=100.0,   # Higher variance for amplitude
        noise=1e-6,
        jitter=1e-6,
        random_state=42,
        n_candidates=5000,
        n_local_restarts=10,
    )
    
    # Run for 25 iterations (should be enough for Branin)
    X, y = opt.run(25)
    
    best_val = -opt.best_y  # Convert back to minimization
    best_x = opt.best_x
    
    print(f"Best found: x={best_x}, f(x)={best_val:.6f}")
    print(f"True minimum: ~0.397887 at (-pi, 12.275), (pi, 2.275), (9.425, 2.475)")
    
    # Check if we found a value close to the global minimum
    # Tolerance: within 0.1 of global min (0.397887)
    global_min = 0.397887
    tolerance = 0.1
    
    success = abs(best_val - global_min) < tolerance
    print(f"Converged within {tolerance}: {success}")
    print(f"Distance to global min: {abs(best_val - global_min):.6f}")
    
    return success, best_val, best_x


def test_kernel_correctness():
    """Test RBF kernel against known values."""
    print("Testing RBF kernel correctness...")
    
    # Test 1: Identical points -> variance
    X = np.array([[1.0, 2.0]])
    K = rbf_kernel(X, X, lengthscale=1.0, variance=2.5)
    assert np.allclose(K[0, 0], 2.5), f"Expected 2.5, got {K[0, 0]}"
    print("  ✓ Identical points give variance")
    
    # Test 2: Symmetry
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    Y = np.array([[5.0, 6.0], [7.0, 8.0]])
    K1 = rbf_kernel(X, Y, lengthscale=2.0, variance=1.0)
    K2 = rbf_kernel(Y, X, lengthscale=2.0, variance=1.0)
    assert np.allclose(K1, K2.T), "Kernel not symmetric"
    print("  ✓ Kernel is symmetric")
    
    # Test 3: Distance-based decay
    X = np.array([[0.0, 0.0]])
    Y1 = np.array([[1.0, 0.0]])  # distance 1
    Y2 = np.array([[2.0, 0.0]])  # distance 2
    K1 = rbf_kernel(X, Y1, lengthscale=1.0, variance=1.0)
    K2 = rbf_kernel(X, Y2, lengthscale=1.0, variance=1.0)
    assert K1 > K2, "Kernel should decay with distance"
    print("  ✓ Kernel decays with distance")
    
    # Test 4: GP fit/predict on simple data
    print("Testing GP fit/predict...")
    X_train = np.array([[0.0], [1.0], [2.0]])
    y_train = np.array([0.0, 1.0, 0.0])  # simple function
    gp = GaussianProcessRegressor(lengthscale=1.0, variance=1.0, noise=1e-4)
    gp.fit(X_train, y_train)
    
    X_test = np.array([[0.5], [1.5]])
    mean, std = gp.predict(X_test)
    assert mean.shape == (2,), f"Mean shape wrong: {mean.shape}"
    assert std.shape == (2,), f"Std shape wrong: {std.shape}"
    assert np.all(std > 0), "Std should be positive"
    print("  ✓ GP fit/predict works")
    
    # Test 5: Acquisition functions
    print("Testing acquisition functions...")
    mean = np.array([0.5, 1.0, 1.5])
    std = np.array([0.2, 0.1, 0.3])
    y_best = 1.0
    
    ei_vals = expected_improvement(mean, std, y_best, xi=0.01)
    assert ei_vals.shape == (3,), f"EI shape wrong: {ei_vals.shape}"
    assert np.all(ei_vals >= 0), "EI should be non-negative"
    
    ucb_vals = upper_confidence_bound(mean, std, beta=2.0)
    assert ucb_vals.shape == (3,), f"UCB shape wrong: {ucb_vals.shape}"
    
    pi_vals = probability_of_improvement(mean, std, y_best, xi=0.01)
    assert pi_vals.shape == (3,), f"PI shape wrong: {pi_vals.shape}"
    assert np.all((pi_vals >= 0) & (pi_vals <= 1)), "PI should be in [0, 1]"
    print("  ✓ Acquisition functions work")
    
    print("All kernel/correctness tests passed!")
    return True


if __name__ == "__main__":
    # Run correctness tests first
    print("=" * 60)
    print("Stage 1: Correctness Tests")
    print("=" * 60)
    test_kernel_correctness()
    
    # Run convergence test
    print("\n" + "=" * 60)
    print("Stage 1 Verification: Branin Convergence Test")
    print("=" * 60)
    success, val, x = test_branin_convergence()
    print("=" * 60)
    if success:
        print("✅ Stage 1 PASSED: Reference BO finds Branin global minimum")
    else:
        print("❌ Stage 1 FAILED: Reference BO did not converge")
    print("=" * 60)