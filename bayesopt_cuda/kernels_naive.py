"""
Python wrapper for naive CUDA kernels.

Provides a clean interface matching the NumPy reference API.
"""

import torch
from typing import Tuple


def _ensure_cuda_tensor(x: torch.Tensor, name: str) -> torch.Tensor:
    """Ensure tensor is on CUDA, float32, contiguous."""
    if not x.is_cuda:
        x = x.cuda()
    if x.dtype != torch.float32:
        x = x.to(dtype=torch.float32)
    if not x.is_contiguous():
        x = x.contiguous()
    return x


def rbf_kernel_naive(
    X: torch.Tensor,
    Y: torch.Tensor,
    lengthscale: float = 1.0,
    variance: float = 1.0,
) -> torch.Tensor:
    """
    Naive CUDA RBF kernel matrix construction.
    
    Args:
        X: (n, d) tensor on CUDA
        Y: (m, d) tensor on CUDA
        lengthscale: kernel lengthscale
        variance: kernel variance
    
    Returns:
        K: (n, m) kernel matrix on CUDA
    """
    from bayesopt_cuda._C import rbf_kernel_naive as _rbf_kernel_naive
    
    X = _ensure_cuda_tensor(X, "X")
    Y = _ensure_cuda_tensor(Y, "Y")
    
    if X.dim() != 2 or Y.dim() != 2:
        raise ValueError("X and Y must be 2D tensors")
    if X.size(1) != Y.size(1):
        raise ValueError("X and Y must have same feature dimension")
    
    return _rbf_kernel_naive(X, Y, lengthscale, variance)


def rbf_kernel_naive_batch(
    X: torch.Tensor,
    Y: torch.Tensor,
    lengthscale: float = 1.0,
    variance: float = 1.0,
) -> torch.Tensor:
    """
    Batched naive CUDA RBF kernel matrix construction.
    
    Args:
        X: (batch, n, d) tensor on CUDA
        Y: (batch, m, d) tensor on CUDA
        lengthscale: kernel lengthscale
        variance: kernel variance
    
    Returns:
        K: (batch, n, m) kernel matrices on CUDA
    """
    from bayesopt_cuda._C import rbf_kernel_naive_batch as _rbf_kernel_naive_batch
    
    X = _ensure_cuda_tensor(X, "X")
    Y = _ensure_cuda_tensor(Y, "Y")
    
    if X.dim() != 3 or Y.dim() != 3:
        raise ValueError("X and Y must be 3D tensors (batch, n, d)")
    if X.size(0) != Y.size(0):
        raise ValueError("X and Y must have same batch size")
    if X.size(2) != Y.size(2):
        raise ValueError("X and Y must have same feature dimension")
    
    return _rbf_kernel_naive_batch(X, Y, lengthscale, variance)


def expected_improvement_naive(
    mean: torch.Tensor,
    std: torch.Tensor,
    y_best: float,
    xi: float = 0.01,
) -> torch.Tensor:
    """
    Naive CUDA Expected Improvement.
    
    Args:
        mean: (m,) predictive mean on CUDA
        std:  (m,) predictive std on CUDA
        y_best: best observed value
        xi: exploration parameter
    
    Returns:
        ei: (m,) expected improvement values
    """
    from bayesopt_cuda._C import expected_improvement_naive as _ei_naive
    
    mean = _ensure_cuda_tensor(mean, "mean")
    std = _ensure_cuda_tensor(std, "std")
    
    if mean.dim() != 1 or std.dim() != 1:
        raise ValueError("mean and std must be 1D tensors")
    if mean.size(0) != std.size(0):
        raise ValueError("mean and std must have same length")
    
    return _ei_naive(mean, std, y_best, xi)


def upper_confidence_bound_naive(
    mean: torch.Tensor,
    std: torch.Tensor,
    beta: float = 2.0,
) -> torch.Tensor:
    """
    Naive CUDA Upper Confidence Bound.
    
    Args:
        mean: (m,) predictive mean on CUDA
        std:  (m,) predictive std on CUDA
        beta: exploration parameter
    
    Returns:
        ucb: (m,) UCB values
    """
    from bayesopt_cuda._C import upper_confidence_bound_naive as _ucb_naive
    
    mean = _ensure_cuda_tensor(mean, "mean")
    std = _ensure_cuda_tensor(std, "std")
    
    if mean.dim() != 1 or std.dim() != 1:
        raise ValueError("mean and std must be 1D tensors")
    if mean.size(0) != std.size(0):
        raise ValueError("mean and std must have same length")
    
    return _ucb_naive(mean, std, beta)


def probability_of_improvement_naive(
    mean: torch.Tensor,
    std: torch.Tensor,
    y_best: float,
    xi: float = 0.01,
) -> torch.Tensor:
    """
    Naive CUDA Probability of Improvement.
    
    Args:
        mean: (m,) predictive mean on CUDA
        std:  (m,) predictive std on CUDA
        y_best: best observed value
        xi: exploration parameter
    
    Returns:
        pi: (m,) PI values
    """
    from bayesopt_cuda._C import probability_of_improvement_naive as _pi_naive
    
    mean = _ensure_cuda_tensor(mean, "mean")
    std = _ensure_cuda_tensor(std, "std")
    
    if mean.dim() != 1 or std.dim() != 1:
        raise ValueError("mean and std must be 1D tensors")
    if mean.size(0) != std.size(0):
        raise ValueError("mean and std must have same length")
    
    return _pi_naive(mean, std, y_best, xi)


# Convenience functions that work with numpy arrays (for testing)
def rbf_kernel_naive_np(
    X: "np.ndarray",
    Y: "np.ndarray",
    lengthscale: float = 1.0,
    variance: float = 1.0,
) -> "np.ndarray":
    """NumPy wrapper for rbf_kernel_naive."""
    import numpy as np
    X_t = torch.from_numpy(X.astype(np.float32)).cuda()
    Y_t = torch.from_numpy(Y.astype(np.float32)).cuda()
    K_t = rbf_kernel_naive(X_t, Y_t, lengthscale, variance)
    return K_t.cpu().numpy()


def expected_improvement_naive_np(
    mean: "np.ndarray",
    std: "np.ndarray",
    y_best: float,
    xi: float = 0.01,
) -> "np.ndarray":
    """NumPy wrapper for expected_improvement_naive."""
    import numpy as np
    mean_t = torch.from_numpy(mean.astype(np.float32)).cuda()
    std_t = torch.from_numpy(std.astype(np.float32)).cuda()
    ei_t = expected_improvement_naive(mean_t, std_t, y_best, xi)
    return ei_t.cpu().numpy()


def upper_confidence_bound_naive_np(
    mean: "np.ndarray",
    std: "np.ndarray",
    beta: float = 2.0,
) -> "np.ndarray":
    """NumPy wrapper for upper_confidence_bound_naive."""
    import numpy as np
    mean_t = torch.from_numpy(mean.astype(np.float32)).cuda()
    std_t = torch.from_numpy(std.astype(np.float32)).cuda()
    ucb_t = upper_confidence_bound_naive(mean_t, std_t, beta)
    return ucb_t.cpu().numpy()


def probability_of_improvement_naive_np(
    mean: "np.ndarray",
    std: "np.ndarray",
    y_best: float,
    xi: float = 0.01,
) -> "np.ndarray":
    """NumPy wrapper for probability_of_improvement_naive."""
    import numpy as np
    mean_t = torch.from_numpy(mean.astype(np.float32)).cuda()
    std_t = torch.from_numpy(std.astype(np.float32)).cuda()
    pi_t = probability_of_improvement_naive(mean_t, std_t, y_best, xi)
    return pi_t.cpu().numpy()


__all__ = [
    "rbf_kernel_naive",
    "rbf_kernel_naive_batch",
    "expected_improvement_naive",
    "upper_confidence_bound_naive",
    "probability_of_improvement_naive",
    "rbf_kernel_naive_np",
    "expected_improvement_naive_np",
    "upper_confidence_bound_naive_np",
    "probability_of_improvement_naive_np",
]