"""
Diagnostic: find the source of the EI mismatch at index 13 for n=10, m=20, d=2.
"""
import numpy as np
import scipy.linalg
import torch

from bayesopt_cuda.reference import rbf_kernel
from bayesopt_cuda.kernels_optimized import rbf_kernel_tiled_np

rng = np.random.default_rng(42)
n, m, d = 10, 20, 2
X_train = rng.normal(size=(n, d)).astype(np.float32)
X_test  = rng.normal(size=(m, d)).astype(np.float32)
Y_train = rng.normal(size=(n,)).astype(np.float32)

lengthscale = 1.2
variance    = 1.5
y_best      = float(np.max(Y_train))
exploration_param = 0.1

# Reference path (NumPy)
K_train = rbf_kernel(X_train, X_train, lengthscale, variance)
K_train[np.diag_indices_from(K_train)] += 1e-6
L    = scipy.linalg.cholesky(K_train, lower=True)
alpha= scipy.linalg.cho_solve((L, True), Y_train).astype(np.float32)
L_inv= scipy.linalg.solve_triangular(L, np.eye(n), lower=True)
K_inv= (L_inv.T @ L_inv).astype(np.float32)

K_cross = rbf_kernel(X_test, X_train, lengthscale, variance)

mean_ref = (K_cross @ alpha).astype(np.float32)
var_ref  = (variance - np.sum((K_cross @ K_inv) * K_cross, axis=1)).astype(np.float32)
var_ref  = np.clip(var_ref, 0, None)
std_ref  = np.sqrt(var_ref).astype(np.float32)

# Focus on index 13
idx = 13
print(f"=== Test point {idx} ===")
print(f"  x_test[{idx}] = {X_test[idx]}")
print(f"  mean_ref  = {mean_ref[idx]:.8f}")
print(f"  std_ref   = {std_ref[idx]:.8f}")
print(f"  var_ref   = {var_ref[idx]:.8f}")

# Now replicate using the tiled RBF to check k_s vector
k_cross_row = rbf_kernel_tiled_np(X_test[idx:idx+1], X_train, lengthscale, variance)[0]
print(f"  k_cross (tiled) = {k_cross_row}")
print(f"  k_cross (numpy) = {K_cross[idx]}")

# Manual computation using kernel k_s
mean_kernel = float(np.dot(k_cross_row, alpha))
var_red      = float(k_cross_row @ K_inv @ k_cross_row)
var_kernel   = variance - var_red
std_kernel   = float(np.sqrt(max(0, var_kernel)))

print(f"  mean_kernel = {mean_kernel:.8f}")
print(f"  std_kernel  = {std_kernel:.8f}")
print(f"  var_kernel  = {var_kernel:.8f}")

# What does acquisition_fused give?
import bayesopt_cuda._C as _C
X_test_t  = torch.from_numpy(X_test).cuda()
X_train_t = torch.from_numpy(X_train).cuda()
alpha_t   = torch.from_numpy(alpha).cuda()
K_inv_t   = torch.from_numpy(K_inv).cuda()
res = _C.acquisition_fused(X_test_t, X_train_t, alpha_t, K_inv_t,
                            float(lengthscale), float(variance),
                            float(y_best), float(exploration_param), 0)
print(f"\n  EI fused output: {res.cpu().numpy()}")

# Reference EI for point 13
from scipy.stats import norm
std_safe = max(std_ref[idx], 1e-12)
z = (mean_ref[idx] - y_best - exploration_param) / std_safe
ei_ref = (mean_ref[idx] - y_best - exploration_param) * norm.cdf(z) + std_safe * norm.pdf(z)
print(f"  EI ref (scipy): {ei_ref:.8f}  (y_best={y_best:.6f})")
