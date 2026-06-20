# BayesOpt-CUDA API Documentation

This document describes the high-level scikit-learn compatible Python classes and the low-level custom CUDA kernel wrappers in **BayesOpt-CUDA**.

---

## 1. High-Level Classes

### 1.1 `GaussianProcessRegressorCUDA`
Exposed in `bayesopt_cuda.gp_cuda`.

#### Constructor
```python
GaussianProcessRegressorCUDA(
    lengthscale: float = 1.0,
    variance: float = 1.0,
    noise: float = 1e-6,
    jitter: float = 1e-6,
    device: str = "cuda"
)
```
* **`lengthscale`**: The lengthscale of the RBF kernel. Controls the smoothness.
* **`variance`**: The output signal variance (amplitude squared) of the RBF kernel.
* **`noise`**: Likelihood noise added to the diagonal of the training covariance matrix for regularisation.
* **`jitter`**: Tiny positive constant added to the diagonal to ensure positive definiteness during Cholesky decomposition.
* **`device`**: Target CUDA device (e.g. `"cuda"`, `"cuda:0"`).

#### Methods

* **`fit(X: np.ndarray, y: np.ndarray) -> self`**:
  Fits the GP model. Computes the covariance matrix $K_{train}$ using the custom tiled RBF kernel, performs the Cholesky decomposition on GPU, and solves for weights $\alpha = K_{train}^{-1} y$.
  * `X`: Array of shape `(n, d)` (training inputs).
  * `y`: Array of shape `(n,)` (training targets).

* **`predict(X: np.ndarray, return_std: bool = True) -> Tuple[np.ndarray, np.ndarray] | np.ndarray`**:
  Predicts mean and standard deviation at test coordinates.
  * `X`: Array of shape `(m, d)`.
  * Returns `(mean, std)` if `return_std` is True, else just `mean`.

* **`acquisition(X_test: np.ndarray, y_best: float, acq_type: str = "ei", exploration_param: float = 0.01) -> np.ndarray`**:
  Evaluates the acquisition function values for candidates. Bypasses matrix materialisation using the fused CUDA kernel.
  * `X_test`: Array of shape `(m, d)`.
  * `y_best`: Best observed value so far.
  * `acq_type`: `"ei"`, `"ucb"`, or `"pi"`.
  * `exploration_param`: $\xi$ for EI/PI, or $\beta$ for UCB.

* **`log_marginal_likelihood() -> float`**:
  Returns the log marginal likelihood of the fitted training dataset on the GPU.

---

### 1.2 `BayesianOptimizerCUDA`
Exposed in `bayesopt_cuda.optimizer_cuda`.

#### Constructor
```python
BayesianOptimizerCUDA(
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
    device: str = "cuda"
)
```
* **`objective`**: Callable to **maximize**. (For minimization, negates values).
* **`bounds`**: Dimension bounds array of shape `(d, 2)`.
* **`n_candidates`**: Number of random grid points to check via fused CUDA acquisition evaluation.
* **`n_local_restarts`**: Number of top candidates to refine using SciPy L-BFGS-B.

#### Methods

* **`step() -> Tuple[np.ndarray, float]`**:
  Runs a single step of the Bayesian Optimization loop. Suggests a new point, evaluates the objective function, and updates the inner GP model.
  * Returns `(x_next, y_next)`.

* **`suggest(n_suggestions: int = 1) -> np.ndarray`**:
  Suggests the next point(s) to query using a Kriging Believer fantasy update strategy if $n > 1$.
  * Returns suggestions array of shape `(n_suggestions, d)`.

* **`run(n_iter: int) -> Tuple[np.ndarray, np.ndarray]`**:
  Runs the optimizer for `n_iter` iterations.
  * Returns `(X_history, y_history)`.

---

## 2. Low-Level Optimized Kernel Functions
Exposed in `bayesopt_cuda.kernels_optimized`.

### 2.1 `rbf_kernel_tiled`
```python
rbf_kernel_tiled(
    X: torch.Tensor,
    Y: torch.Tensor,
    lengthscale: float = 1.0,
    variance: float = 1.0
) -> torch.Tensor
```
* Custom tiled float32 RBF kernel. Expects contiguous CUDA tensors.
* Returns `K` of shape `(X.shape[0], Y.shape[0])` in float32.

### 2.2 `rbf_kernel_tiled_fp16`
```python
rbf_kernel_tiled_fp16(
    X: torch.Tensor,
    Y: torch.Tensor,
    lengthscale: float = 1.0,
    variance: float = 1.0
) -> torch.Tensor
```
* FP16 mixed precision RBF kernel.
* `X`: contiguous `torch.float16` CUDA tensor of shape `(n, d)`.
* `Y`: contiguous `torch.float16` CUDA tensor of shape `(m, d)`.
* Returns `K` of shape `(n, m)` in `torch.float32`.

### 2.3 `acquisition_fused`
```python
acquisition_fused(
    X_test: torch.Tensor,
    X_train: torch.Tensor,
    alpha: torch.Tensor,
    K_inv: torch.Tensor,
    lengthscale: float = 1.0,
    variance: float = 1.0,
    y_best: float = 0.0,
    exploration_param: float = 0.01,
    acq_type: str = "ei"
) -> torch.Tensor
```
* Fused acquisition evaluation using pre-inverted covariance matrix `K_inv`.
* Limit $n \le 64$ due to $O(n^2)$ shared memory limitations.

### 2.4 `acquisition_fused_chol`
```python
acquisition_fused_chol(
    X_test: torch.Tensor,
    X_train: torch.Tensor,
    alpha: torch.Tensor,
    L: torch.Tensor,
    lengthscale: float = 1.0,
    variance: float = 1.0,
    y_best: float = 0.0,
    exploration_param: float = 0.01,
    acq_type: str = "ei"
) -> torch.Tensor
```
* Fused acquisition evaluation using Cholesky factor `L`.
* Utilises in-kernel forward solve. Relies on $O(n)$ shared memory, allowing scaling to arbitrary numbers of training points.
