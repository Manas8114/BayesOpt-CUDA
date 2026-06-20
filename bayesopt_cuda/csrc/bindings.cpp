/**
 * Pybind11 bindings for BayesOpt-CUDA naive kernels
 * 
 * Exposes:
 *   - rbf_kernel_naive(X, Y, lengthscale, variance) -> K
 *   - expected_improvement_naive(mean, std, y_best, xi) -> ei
 *   - upper_confidence_bound_naive(mean, std, beta) -> ucb
 *   - probability_of_improvement_naive(mean, std, y_best, xi) -> pi
 * 
 * All inputs/outputs are torch.Tensor on CUDA device, float32.
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <stdexcept>

// Forward declarations from kernel files
void rbf_kernel_naive_launch(
    const float* X, const float* Y, float* K,
    const int n, const int m, const int d,
    const float lengthscale, const float variance
);

void rbf_kernel_naive_batch_launch(
    const float* X, const float* Y, float* K,
    const int batch, const int n, const int m, const int d,
    const float lengthscale, const float variance
);

void expected_improvement_naive_launch(
    const float* mean, const float* std, float* ei,
    const int m, const float y_best, const float xi
);

void upper_confidence_bound_naive_launch(
    const float* mean, const float* std, float* ucb,
    const int m, const float beta
);

void probability_of_improvement_naive_launch(
    const float* mean, const float* std, float* pi,
    const int m, const float y_best, const float xi
);

// Forward declarations for Optimized kernels
void launch_rbf_kernel_tiled(
    const float* X, const float* Y, float* K,
    const int n, const int m, const int d,
    const float lengthscale, const float variance
);

void launch_acquisition_fused(
    const float* X_test, const float* X_train,
    const float* alpha, const float* K_inv,
    float* out, const int m, const int n, const int d,
    const float lengthscale, const float variance,
    const float y_best, const float exploration_param,
    const int acq_type_int
);

// Forward declarations for v2 optimised kernels
void launch_acquisition_fused_chol(
    const float* X_test, const float* X_train,
    const float* alpha, const float* L,
    float* out, const int m, const int n, const int d,
    const float lengthscale, const float variance,
    const float y_best, const float exploration_param,
    const int acq_type_int
);

void launch_rbf_kernel_tiled_fp16(
    const __half* X, const __half* Y, float* K,
    const int n, const int m, const int d,
    const float lengthscale, const float variance
);

// Helper to validate tensor properties
inline void check_cuda_tensor(const torch::Tensor& t, const char* name, int expected_ndim = -1) {
    if (!t.is_cuda()) {
        throw std::runtime_error(std::string(name) + " must be a CUDA tensor");
    }
    if (t.scalar_type() != torch::kFloat32) {
        throw std::runtime_error(std::string(name) + " must be float32");
    }
    if (expected_ndim >= 0 && t.dim() != expected_ndim) {
        throw std::runtime_error(
            std::string(name) + " must be " + std::to_string(expected_ndim) + "D, got " + std::to_string(t.dim())
        );
    }
    if (!t.is_contiguous()) {
        throw std::runtime_error(std::string(name) + " must be contiguous");
    }
}

inline void check_fp16_tensor(const torch::Tensor& t, const char* name, int expected_ndim = -1) {
    if (!t.is_cuda()) {
        throw std::runtime_error(std::string(name) + " must be a CUDA tensor");
    }
    if (t.scalar_type() != torch::kFloat16) {
        throw std::runtime_error(std::string(name) + " must be float16");
    }
    if (expected_ndim >= 0 && t.dim() != expected_ndim) {
        throw std::runtime_error(
            std::string(name) + " must be " + std::to_string(expected_ndim) + "D, got " + std::to_string(t.dim())
        );
    }
    if (!t.is_contiguous()) {
        throw std::runtime_error(std::string(name) + " must be contiguous");
    }
}


// rbf_kernel_naive(X, Y, lengthscale, variance) -> K
torch::Tensor rbf_kernel_naive(
    torch::Tensor X,
    torch::Tensor Y,
    float lengthscale,
    float variance
) {
    check_cuda_tensor(X, "X", 2);
    check_cuda_tensor(Y, "Y", 2);
    
    const int n = X.size(0);
    const int d = X.size(1);
    const int m = Y.size(0);
    const int d2 = Y.size(1);
    
    if (d != d2) {
        throw std::runtime_error("X and Y must have same feature dimension");
    }
    
    // Output tensor
    torch::Tensor K = torch::empty({n, m}, X.options());
    
    // Launch kernel
    rbf_kernel_naive_launch(
        X.data_ptr<float>(),
        Y.data_ptr<float>(),
        K.data_ptr<float>(),
        n, m, d,
        lengthscale, variance
    );
    
    // Synchronize to catch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("rbf_kernel_naive kernel failed: ") + cudaGetErrorString(err));
    }
    
    return K;
}


// rbf_kernel_naive_batch(X, Y, lengthscale, variance) -> K
// X: (batch, n, d), Y: (batch, m, d) -> K: (batch, n, m)
torch::Tensor rbf_kernel_naive_batch(
    torch::Tensor X,
    torch::Tensor Y,
    float lengthscale,
    float variance
) {
    check_cuda_tensor(X, "X", 3);
    check_cuda_tensor(Y, "Y", 3);
    
    const int batch = X.size(0);
    const int n = X.size(1);
    const int d = X.size(2);
    const int batch2 = Y.size(0);
    const int m = Y.size(1);
    const int d2 = Y.size(2);
    
    if (batch != batch2) {
        throw std::runtime_error("X and Y must have same batch size");
    }
    if (d != d2) {
        throw std::runtime_error("X and Y must have same feature dimension");
    }
    
    torch::Tensor K = torch::empty({batch, n, m}, X.options());
    
    rbf_kernel_naive_batch_launch(
        X.data_ptr<float>(),
        Y.data_ptr<float>(),
        K.data_ptr<float>(),
        batch, n, m, d,
        lengthscale, variance
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("rbf_kernel_naive_batch kernel failed: ") + cudaGetErrorString(err));
    }
    
    return K;
}


// expected_improvement_naive(mean, std, y_best, xi) -> ei
torch::Tensor expected_improvement_naive(
    torch::Tensor mean,
    torch::Tensor std,
    float y_best,
    float xi
) {
    check_cuda_tensor(mean, "mean", 1);
    check_cuda_tensor(std, "std", 1);
    
    const int m = mean.size(0);
    if (std.size(0) != m) {
        throw std::runtime_error("mean and std must have same length");
    }
    
    torch::Tensor ei = torch::empty({m}, mean.options());
    
    expected_improvement_naive_launch(
        mean.data_ptr<float>(),
        std.data_ptr<float>(),
        ei.data_ptr<float>(),
        m, y_best, xi
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("expected_improvement_naive kernel failed: ") + cudaGetErrorString(err));
    }
    
    return ei;
}


// upper_confidence_bound_naive(mean, std, beta) -> ucb
torch::Tensor upper_confidence_bound_naive(
    torch::Tensor mean,
    torch::Tensor std,
    float beta
) {
    check_cuda_tensor(mean, "mean", 1);
    check_cuda_tensor(std, "std", 1);
    
    const int m = mean.size(0);
    if (std.size(0) != m) {
        throw std::runtime_error("mean and std must have same length");
    }
    
    torch::Tensor ucb = torch::empty({m}, mean.options());
    
    upper_confidence_bound_naive_launch(
        mean.data_ptr<float>(),
        std.data_ptr<float>(),
        ucb.data_ptr<float>(),
        m, beta
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("upper_confidence_bound_naive kernel failed: ") + cudaGetErrorString(err));
    }
    
    return ucb;
}


// probability_of_improvement_naive(mean, std, y_best, xi) -> pi
torch::Tensor probability_of_improvement_naive(
    torch::Tensor mean,
    torch::Tensor std,
    float y_best,
    float xi
) {
    check_cuda_tensor(mean, "mean", 1);
    check_cuda_tensor(std, "std", 1);
    
    const int m = mean.size(0);
    if (std.size(0) != m) {
        throw std::runtime_error("mean and std must have same length");
    }
    
    torch::Tensor pi = torch::empty({m}, mean.options());
    
    probability_of_improvement_naive_launch(
        mean.data_ptr<float>(),
        std.data_ptr<float>(),
        pi.data_ptr<float>(),
        m, y_best, xi
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("probability_of_improvement_naive kernel failed: ") + cudaGetErrorString(err));
    }
    
    return pi;
}

// rbf_kernel_tiled(X, Y, lengthscale, variance) -> K
torch::Tensor rbf_kernel_tiled(
    torch::Tensor X,
    torch::Tensor Y,
    float lengthscale,
    float variance
) {
    check_cuda_tensor(X, "X", 2);
    check_cuda_tensor(Y, "Y", 2);
    
    const int n = X.size(0);
    const int d = X.size(1);
    const int m = Y.size(0);
    const int d2 = Y.size(1);
    
    if (d != d2) {
        throw std::runtime_error("X and Y must have same feature dimension");
    }
    
    // Output tensor
    torch::Tensor K = torch::empty({n, m}, X.options());
    
    // Launch kernel
    launch_rbf_kernel_tiled(
        X.data_ptr<float>(),
        Y.data_ptr<float>(),
        K.data_ptr<float>(),
        n, m, d,
        lengthscale, variance
    );
    
    // Synchronize to catch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("rbf_kernel_tiled kernel failed: ") + cudaGetErrorString(err));
    }
    
    return K;
}

// acquisition_fused(X_test, X_train, alpha, K_inv, lengthscale, variance, y_best, exploration_param, acq_type) -> acq_values
torch::Tensor acquisition_fused(
    torch::Tensor X_test,
    torch::Tensor X_train,
    torch::Tensor alpha,
    torch::Tensor K_inv,
    float lengthscale,
    float variance,
    float y_best,
    float exploration_param,
    int acq_type
) {
    check_cuda_tensor(X_test, "X_test", 2);
    check_cuda_tensor(X_train, "X_train", 2);
    check_cuda_tensor(alpha, "alpha", 1);
    check_cuda_tensor(K_inv, "K_inv", 2);

    const int m = X_test.size(0);
    const int d = X_test.size(1);
    const int n = X_train.size(0);
    
    if (X_train.size(1) != d) throw std::runtime_error("X_test and X_train must have same feature dimension");
    if (alpha.size(0) != n) throw std::runtime_error("alpha must have length equal to number of training points");
    if (K_inv.size(0) != n || K_inv.size(1) != n) throw std::runtime_error("K_inv must be n x n");

    torch::Tensor out = torch::empty({m}, X_test.options());

    launch_acquisition_fused(
        X_test.data_ptr<float>(),
        X_train.data_ptr<float>(),
        alpha.data_ptr<float>(),
        K_inv.data_ptr<float>(),
        out.data_ptr<float>(),
        m, n, d,
        lengthscale, variance,
        y_best, exploration_param,
        acq_type
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("acquisition_fused kernel failed: ") + cudaGetErrorString(err));
    }

    return out;
}


// acquisition_fused_chol: uses Cholesky factor L instead of K_inv → O(n) SMEM
torch::Tensor acquisition_fused_chol(
    torch::Tensor X_test,
    torch::Tensor X_train,
    torch::Tensor alpha,
    torch::Tensor L,
    float lengthscale,
    float variance,
    float y_best,
    float exploration_param,
    int acq_type
) {
    check_cuda_tensor(X_test, "X_test", 2);
    check_cuda_tensor(X_train, "X_train", 2);
    check_cuda_tensor(alpha, "alpha", 1);
    check_cuda_tensor(L, "L", 2);

    const int m = X_test.size(0);
    const int d = X_test.size(1);
    const int n = X_train.size(0);

    if (X_train.size(1) != d) throw std::runtime_error("X_test and X_train must have same feature dimension");
    if (alpha.size(0) != n)   throw std::runtime_error("alpha must have length n");
    if (L.size(0) != n || L.size(1) != n) throw std::runtime_error("L must be n x n");

    torch::Tensor out = torch::empty({m}, X_test.options());

    launch_acquisition_fused_chol(
        X_test.data_ptr<float>(),
        X_train.data_ptr<float>(),
        alpha.data_ptr<float>(),
        L.data_ptr<float>(),
        out.data_ptr<float>(),
        m, n, d,
        lengthscale, variance,
        y_best, exploration_param,
        acq_type
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("acquisition_fused_chol kernel failed: ") + cudaGetErrorString(err));

    return out;
}

// rbf_kernel_tiled_fp16: float16 inputs, float32 output K
torch::Tensor rbf_kernel_tiled_fp16(
    torch::Tensor X,
    torch::Tensor Y,
    float lengthscale,
    float variance
) {
    check_fp16_tensor(X, "X", 2);
    check_fp16_tensor(Y, "Y", 2);

    const int n  = X.size(0);
    const int d  = X.size(1);
    const int m  = Y.size(0);
    const int d2 = Y.size(1);

    if (d != d2) throw std::runtime_error("X and Y must have same feature dimension");

    // Output is float32
    torch::Tensor K = torch::empty({n, m},
        X.options().dtype(torch::kFloat32));

    launch_rbf_kernel_tiled_fp16(
        reinterpret_cast<const __half*>(X.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(Y.data_ptr<at::Half>()),
        K.data_ptr<float>(),
        n, m, d,
        lengthscale, variance
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("rbf_kernel_tiled_fp16 kernel failed: ") + cudaGetErrorString(err));

    return K;
}


// Module definition
PYBIND11_MODULE(_C, m) {
    m.doc() = "BayesOpt-CUDA: CUDA Kernels for Bayesian Optimization";
    
    m.def("rbf_kernel_naive", &rbf_kernel_naive,
        "Naive RBF kernel matrix construction",
        py::arg("X"), py::arg("Y"), py::arg("lengthscale"), py::arg("variance")
    );
    
    m.def("rbf_kernel_naive_batch", &rbf_kernel_naive_batch,
        "Batched naive RBF kernel matrix construction",
        py::arg("X"), py::arg("Y"), py::arg("lengthscale"), py::arg("variance")
    );
    
    m.def("expected_improvement_naive", &expected_improvement_naive,
        "Naive Expected Improvement acquisition function",
        py::arg("mean"), py::arg("std"), py::arg("y_best"), py::arg("xi")
    );
    
    m.def("upper_confidence_bound_naive", &upper_confidence_bound_naive,
        "Naive Upper Confidence Bound acquisition function",
        py::arg("mean"), py::arg("std"), py::arg("beta")
    );
    
    m.def("probability_of_improvement_naive", &probability_of_improvement_naive,
        "Naive Probability of Improvement acquisition function",
        py::arg("mean"), py::arg("std"), py::arg("y_best"), py::arg("xi")
    );
    
    m.def("rbf_kernel_tiled", &rbf_kernel_tiled,
        "Tiled RBF kernel matrix construction (float32)",
        py::arg("X"), py::arg("Y"), py::arg("lengthscale"), py::arg("variance")
    );

    m.def("acquisition_fused", &acquisition_fused,
        "Fused acquisition (EI/UCB/PI) — uses K_inv, fast for n<=64",
        py::arg("X_test"), py::arg("X_train"), py::arg("alpha"), py::arg("K_inv"),
        py::arg("lengthscale"), py::arg("variance"), py::arg("y_best"), py::arg("exploration_param"),
        py::arg("acq_type")
    );

    m.def("acquisition_fused_chol", &acquisition_fused_chol,
        "Fused acquisition (EI/UCB/PI) — uses Cholesky factor L, O(n) SMEM, works for any n",
        py::arg("X_test"), py::arg("X_train"), py::arg("alpha"), py::arg("L"),
        py::arg("lengthscale"), py::arg("variance"), py::arg("y_best"), py::arg("exploration_param"),
        py::arg("acq_type")
    );

    m.def("rbf_kernel_tiled_fp16", &rbf_kernel_tiled_fp16,
        "Tiled RBF kernel with float16 inputs and float32 output (2x memory bandwidth)",
        py::arg("X"), py::arg("Y"), py::arg("lengthscale"), py::arg("variance")
    );
}