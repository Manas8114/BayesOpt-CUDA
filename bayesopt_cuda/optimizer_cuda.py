"""
CUDA-accelerated Bayesian Optimizer.

A drop-in replacement for the reference BayesianOptimizer, but using
GaussianProcessRegressorCUDA under the hood for kernel matrix construction
and acquisition function evaluation.

Example
-------
>>> from bayesopt_cuda.optimizer_cuda import BayesianOptimizerCUDA
>>> opt = BayesianOptimizerCUDA(
...     objective=lambda x: -branin(x),
...     bounds=np.array([[-5, 10], [0, 15]]),
...     acquisition="ei",
... )
>>> X, y = opt.run(n_iter=20)
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from .gp_cuda import GaussianProcessRegressorCUDA


class BayesianOptimizerCUDA:
    """
    GPU-accelerated Bayesian Optimization loop.

    Parameters
    ----------
    objective : callable
        Function to *maximize* (takes 1D ndarray, returns float).
        For minimization, pass the negated function.
    bounds : ndarray, shape (d, 2)
        Lower and upper bounds for each dimension.
    acquisition : str
        "ei", "ucb", or "pi".
    lengthscale : float
        RBF kernel lengthscale.
    variance : float
        RBF kernel variance.
    noise : float
        Observation noise.
    jitter : float
        Cholesky stability jitter.
    xi : float
        Exploration parameter for EI/PI.
    beta : float
        Exploration parameter for UCB (UCB = mean + sqrt(beta) * std).
    random_state : int, optional
        Seed for reproducibility.
    n_candidates : int
        Random candidates for acquisition optimization.
    n_local_restarts : int
        Number of top candidates to refine with L-BFGS-B.
    device : str
        CUDA device string.
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
        device: str = "cuda",
    ):
        self.objective = objective
        self.bounds = np.asarray(bounds, dtype=np.float32)
        self.acquisition_name = acquisition
        self.lengthscale = lengthscale
        self.variance = variance
        self.noise = noise
        self.jitter = jitter
        self.xi = xi
        self.beta = beta
        self.n_candidates = n_candidates
        self.n_local_restarts = n_local_restarts
        self.device = device

        if random_state is not None:
            np.random.seed(random_state)

        self.X_: List[np.ndarray] = []
        self.y_: List[float] = []
        self.gp_: Optional[GaussianProcessRegressorCUDA] = None
        self.n_iter_ = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _new_gp(self) -> GaussianProcessRegressorCUDA:
        return GaussianProcessRegressorCUDA(
            lengthscale=self.lengthscale,
            variance=self.variance,
            noise=self.noise,
            jitter=self.jitter,
            device=self.device,
        )

    def _acq_exploration_param(self) -> float:
        if self.acquisition_name == "ucb":
            return self.beta
        return self.xi

    def _optimize_acquisition(self) -> np.ndarray:
        """Optimize acquisition via random sampling + L-BFGS-B refinement."""
        d = self.bounds.shape[0]
        y_best = float(np.max(self.y_))

        # Random sampling
        X_cand = np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1],
            size=(self.n_candidates, d),
        ).astype(np.float32)

        # Evaluate acquisition (batch via CUDA fused kernel)
        acq_vals = self.gp_.acquisition(
            X_cand, y_best,
            acq_type=self.acquisition_name,
            exploration_param=self._acq_exploration_param(),
        )

        # Take top candidates
        n_top = min(self.n_local_restarts, self.n_candidates)
        top_idx = np.argsort(acq_vals)[-n_top:][::-1]
        X_top = X_cand[top_idx]

        # L-BFGS-B refinement
        best_x = X_top[0]
        best_acq = acq_vals[top_idx[0]]

        for x0 in X_top:
            def neg_acq(x: np.ndarray) -> float:
                x32 = np.asarray(x, dtype=np.float32).reshape(1, -1)
                vals = self.gp_.acquisition(
                    x32, y_best,
                    acq_type=self.acquisition_name,
                    exploration_param=self._acq_exploration_param(),
                )
                return float(-vals[0])

            res = minimize(
                neg_acq, x0,
                bounds=self.bounds.tolist(),
                method="L-BFGS-B",
                options={"maxiter": 50},
            )
            if res.success and (-res.fun) > best_acq:
                best_acq = -res.fun
                best_x = res.x

        return best_x.astype(np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def suggest(self, n_suggestions: int = 1) -> np.ndarray:
        """Suggest next point(s) to evaluate."""
        if self.gp_ is None:
            d = self.bounds.shape[0]
            return np.random.uniform(
                self.bounds[:, 0], self.bounds[:, 1],
                size=(n_suggestions, d),
            ).astype(np.float32)

        suggestions = []
        for _ in range(n_suggestions):
            x_next = self._optimize_acquisition()
            suggestions.append(x_next)
            # Fantasy update: temporarily add to GP
            mean, _ = self.gp_.predict(x_next.reshape(1, -1), return_std=True)
            X_temp = np.vstack(self.X_ + [x_next])
            y_temp = np.append(self.y_, mean[0])
            self.gp_ = self._new_gp().fit(X_temp, y_temp)

        # Restore original GP
        self.gp_ = self._new_gp().fit(np.array(self.X_), np.array(self.y_))
        return np.array(suggestions)

    def step(self) -> Tuple[np.ndarray, float]:
        """Perform one BO iteration: suggest → evaluate → update GP."""
        x_next = self.suggest(1)[0]
        y_next = self.objective(x_next)

        self.X_.append(x_next)
        self.y_.append(float(y_next))

        self.gp_ = self._new_gp().fit(np.array(self.X_), np.array(self.y_))
        self.n_iter_ += 1
        return x_next, y_next

    def run(self, n_iter: int) -> Tuple[np.ndarray, np.ndarray]:
        """Run the BO loop for n_iter iterations."""
        for _ in range(n_iter):
            self.step()
        return np.array(self.X_), np.array(self.y_)

    @property
    def best_x(self) -> np.ndarray:
        if not self.y_:
            raise ValueError("No evaluations yet")
        return self.X_[int(np.argmax(self.y_))]

    @property
    def best_y(self) -> float:
        if not self.y_:
            raise ValueError("No evaluations yet")
        return float(np.max(self.y_))
