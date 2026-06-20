"""
CUDA-accelerated Gaussian Process Regressor.

sklearn-compatible estimator that uses our custom CUDA kernels for
kernel matrix construction (tiled RBF) and acquisition function evaluation
(fused EI/UCB/PI).

API mirrors sklearn's GaussianProcessRegressor:
    gp = GaussianProcessRegressorCUDA(lengthscale=1.0, variance=1.0)
    gp.fit(X_train, y_train)
    mean, std = gp.predict(X_test)

Internally:
  - Kernel matrix built with CUDA tiled RBF (Stage 3)
  - Cholesky + solve via torch.linalg (GPU)
  - Acquisition eval with fused CUDA kernel (Stage 3)
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np
import torch

from .kernels_optimized import rbf_kernel_tiled, acquisition_fused


class GaussianProcessRegressorCUDA:
    """
    GPU-accelerated GP Regressor with RBF kernel.

    Parameters
    ----------
    lengthscale : float
        RBF lengthscale.
    variance : float
        RBF signal variance (amplitude squared).
    noise : float
        Observation noise (added to diagonal of K_train).
    jitter : float
        Numerical jitter for Cholesky stability.
    device : str
        CUDA device string, e.g. ``"cuda:0"``.
    """

    def __init__(
        self,
        lengthscale: float = 1.0,
        variance: float = 1.0,
        noise: float = 1e-6,
        jitter: float = 1e-6,
        device: str = "cuda",
    ):
        self.lengthscale = float(lengthscale)
        self.variance = float(variance)
        self.noise = float(noise)
        self.jitter = float(jitter)
        self.device = torch.device(device)

        self._X_train: Optional[torch.Tensor] = None
        self._y_train: Optional[torch.Tensor] = None
        self._L: Optional[torch.Tensor] = None        # Cholesky factor
        self._alpha: Optional[torch.Tensor] = None    # K^{-1} y
        self._K_inv: Optional[torch.Tensor] = None    # K^{-1}  (for variance)
        self.is_fitted_ = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_tensor(self, X: np.ndarray, dtype=torch.float32) -> torch.Tensor:
        return torch.as_tensor(X, dtype=dtype, device=self.device).contiguous()

    def _kernel(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        return rbf_kernel_tiled(X, Y, self.lengthscale, self.variance)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianProcessRegressorCUDA":
        """
        Fit the GP to training data.

        Parameters
        ----------
        X : array-like, shape (n, d)
        y : array-like, shape (n,)

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()

        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got {X.shape}")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError("y must be 1D with same length as X")

        self._X_train = self._to_tensor(X)
        self._y_train = self._to_tensor(y)
        n = X.shape[0]

        # Kernel matrix
        K = self._kernel(self._X_train, self._X_train)  # (n, n)

        # Add noise + jitter to diagonal
        diag_val = self.noise + self.jitter
        K = K + diag_val * torch.eye(n, dtype=torch.float32, device=self.device)

        # Cholesky decomposition
        try:
            self._L = torch.linalg.cholesky(K)
        except Exception as e:
            warnings.warn(f"Cholesky failed ({e}), adding more jitter")
            K = K + 1e-4 * torch.eye(n, dtype=torch.float32, device=self.device)
            self._L = torch.linalg.cholesky(K)

        # alpha = K^{-1} y
        # Solve L @ v = y, then L^T @ alpha = v
        self._alpha = torch.cholesky_solve(
            self._y_train.unsqueeze(-1), self._L
        ).squeeze(-1)

        # K_inv = L^{-T} L^{-1}  (needed for variance computation)
        L_inv = torch.linalg.solve_triangular(
            self._L, torch.eye(n, dtype=torch.float32, device=self.device), upper=False
        )
        self._K_inv = L_inv.T @ L_inv

        self.is_fitted_ = True
        return self

    def predict(
        self,
        X: np.ndarray,
        return_std: bool = True,
    ) -> Tuple[np.ndarray, ...]:
        """
        Predict at test points.

        Parameters
        ----------
        X : array-like, shape (m, d)
        return_std : bool
            If True, also return predictive std.

        Returns
        -------
        mean : ndarray, shape (m,)
        std  : ndarray, shape (m,)   [only if return_std=True]
        """
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before predict()")

        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_t = self._to_tensor(X)

        K_cross = self._kernel(X_t, self._X_train)   # (m, n)

        # Mean
        mean = K_cross @ self._alpha                  # (m,)

        if not return_std:
            return mean.cpu().numpy()

        # Variance: sigma^2 = k(x,x) - k_cross @ K_inv @ k_cross^T
        # Since k(x,x) = variance for RBF, per-point we have:
        #   var[i] = variance - k_cross[i] @ K_inv @ k_cross[i]
        V = K_cross @ self._K_inv             # (m, n)
        var = self.variance - (V * K_cross).sum(dim=1)  # (m,)
        var = torch.clamp(var, min=0.0)
        std = torch.sqrt(var)

        return mean.cpu().numpy(), std.cpu().numpy()

    def acquisition(
        self,
        X_test: np.ndarray,
        y_best: float,
        acq_type: str = "ei",
        exploration_param: float = 0.01,
    ) -> np.ndarray:
        """
        Evaluate the acquisition function at candidate points.

        Uses the fused CUDA kernel for EI, UCB, and PI.

        Parameters
        ----------
        X_test : array-like, shape (m, d)
        y_best : float
            Best observed value (for EI/PI).
        acq_type : str
            One of "ei", "ucb", "pi".
        exploration_param : float
            xi (for EI/PI) or beta (for UCB — passed as-is, i.e. UCB = mu + sqrt(beta)*std).

        Returns
        -------
        acq_values : ndarray, shape (m,)
        """
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before acquisition()")

        X_test = np.asarray(X_test, dtype=np.float32)
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)
        X_test_t = self._to_tensor(X_test)

        return acquisition_fused(
            X_test_t,
            self._X_train,
            self._alpha,
            self._K_inv,
            lengthscale=self.lengthscale,
            variance=self.variance,
            y_best=float(y_best),
            exploration_param=float(exploration_param),
            acq_type=acq_type,
        ).cpu().numpy()

    def log_marginal_likelihood(self) -> float:
        """Compute log marginal likelihood of training data."""
        if not self.is_fitted_:
            raise RuntimeError("Call fit() first")
        n = self._X_train.shape[0]
        # log p(y|X) = -0.5 * y^T K^{-1} y - sum(log diag(L)) - n/2 log(2pi)
        yT_alpha = (self._y_train * self._alpha).sum()
        logdet = 2.0 * self._L.diagonal().log().sum()
        lml = -0.5 * yT_alpha - logdet - 0.5 * n * torch.log(torch.tensor(2 * np.pi))
        return float(lml.item())
