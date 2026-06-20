# BayesOpt-CUDA Architecture Documentation

This document describes the architectural layout, component interactions, data flows, and hardware utilisation of **BayesOpt-CUDA**.

---

## 1. System Overview & Component Hierarchy

The library is split into two primary layers:
1. **Python User Interface Layer**: Scikit-Learn / BoTorch-compatible estimators that manage Gaussian Process fitting, state, objective loops, and local acquisition function refinement (via L-BFGS-B).
2. **CUDA Core Performance Layer**: Custom compiled C++/CUDA kernels performing distance computation, kernel matrix tiling, and multi-candidate fused acquisition evaluations.

The components interact as follows:

```mermaid
graph TD
    subgraph Python API (Host)
        BO[BayesianOptimizerCUDA] -->|"Calls fit() & predict()"| GP[GaussianProcessRegressorCUDA]
        GP -->|"Calls kernel & acquisition wrappers"| KO[bayesopt_cuda.kernels_optimized]
    end

    subgraph PyBind11 C++ Module
        KO -->|"Loads/Calls"| PB[bayesopt_cuda._C bindings.cpp]
    end

    subgraph CUDA Device Kernels
        PB -->|"Launches"| RK_naive[rbf_kernel_naive.cu]
        PB -->|"Launches"| RK_tiled[rbf_kernel_tiled.cu]
        PB -->|"Launches"| RK_fp16[rbf_kernel_tiled_fp16.cu]
        PB -->|"Launches"| EI_naive[ei_kernel_naive.cu]
        PB -->|"Launches"| ACQ_fused[acquisition_fused.cu]
        PB -->|"Launches"| ACQ_chol[acquisition_fused_cholesky.cu]
    end

    subgraph GPU Hardware Execution
        RK_fp16 -->|"FP16 SIMD __half2 & DRAM reduction"| HW_FP16[RTX Tensor Cores / ALU]
        ACQ_chol -->|"O(n) SMEM Forward-Substitution"| HW_SMEM[Shared Memory L1 Cache]
    end
```

---

## 2. Component Reference

### 2.1 Python API (`bayesopt_cuda`)
* **[GaussianProcessRegressorCUDA](file:///c:/Users/msgok/OneDrive/Desktop/Project/Hermes/bayesopt_cuda/bayesopt_cuda/gp_cuda.py)**: Maintains training data `X_train` and `y_train` on host/device. Performs the global Cholesky decomposition of the covariance matrix ($K_{train} = L L^T$) on the GPU using `torch.linalg.cholesky`.
* **[BayesianOptimizerCUDA](file:///c:/Users/msgok/OneDrive/Desktop/Project/Hermes/bayesopt_cuda/bayesopt_cuda/optimizer_cuda.py)**: Manages the Bayesian Optimization loop. It samples high-dimensional candidate grids on the GPU, evaluates them concurrently using the fused acquisition kernel, and refines the top restart locations using PySciPy L-BFGS-B.
* **[kernels_optimized](file:///c:/Users/msgok/OneDrive/Desktop/Project/Hermes/bayesopt_cuda/bayesopt_cuda/kernels_optimized.py)**: A Python translation layer exposing the PyBind11 extension functions. Handles numpy-to-tensor conversions, contiguous memory checking, and parameter bounds check.

### 2.2 Compilation & Bindings (`bayesopt_cuda._C`)
* **[bindings.cpp](file:///c:/Users/msgok/OneDrive/Desktop/Project/Hermes/bayesopt_cuda/bayesopt_cuda/csrc/bindings.cpp)**: Binds the C++ and CUDA compilation units to Python using PyBind11. It checks tensor device (must be CUDA), alignment (must be contiguous), and sizes, and extracts raw device pointers (`float*`, `__half*`) to pass into the CUDA thread dispatchers.
* **JIT vs AOT**: Exposes `load_extension()` in the package namespace. If the module is not pre-compiled, it JIT-compiles all C++/CUDA sources on-the-fly and caches the output under `~/.cache/torch_extensions`.

### 2.3 CUDA Implementations (`csrc`)
* **`rbf_kernel_tiled.cu`**: FP32 tiled RBF. Amortises DRAM reads by staging $16 \times 16$ blocks of training and test coordinates into Shared Memory (SMEM).
* **`rbf_kernel_tiled_fp16.cu`**: Mixed precision RBF. Loads data from DRAM as FP16 (halving bandwidth), processes distances in registers using `__half2` SIMD instructions, and converts to FP32 only for `expf()` evaluation to preserve numeric range and accuracy.
* **`acquisition_fused.cu`**: Performs fused acquisition evaluation in $O(n^2)$ SMEM by storing the pre-inverted covariance matrix ($K^{-1}$).
* **`acquisition_fused_cholesky.cu`**: Solves the lower-triangular Cholesky factor $L$ in-kernel ($Lv = k_s$) using a forward substitution with $O(n)$ SMEM complexity, permitting scaling to arbitrary numbers of training points.

---

## 3. Data Flow Diagrams

### 3.1 Fitting the Gaussian Process
The data flow for model fitting runs as follows:

```
[numpy X, y] ──> [torch CUDA float32]
                      │
                      ▼
         [rbf_kernel_tiled] (CUDA) ──> [Covariance Matrix K (n x n)]
                                                     │
                                                     ▼
                                      [torch.linalg.cholesky]
                                                     │
                                                     ▼
                                        [Cholesky Factor L (n x n)]
                                                     │
                                                     ▼
                                        [torch.cholesky_solve]
                                                     │
                                                     ▼
                                            [alpha (n x 1)]
```

### 3.2 Fused Acquisition Evaluation (Expected Improvement)
The data flow for acquisition evaluation bypasses allocating intermediate covariance matrices ($m \times n$) in global memory:

```
[Candidate Points X_test (m x d)] ────┐
                                     ├─> [acquisition_fused_cholesky] ─> [Acquisition Values (m)]
[Training Points X_train (n x d)] ───┤
[Cholesky L (n x n)] ────────────────┤
[alpha (n x 1)] ─────────────────────┘
```
Inside the kernel:
1. Each block processes a subset of candidates.
2. For each candidate, $k_{cross}$ (RBF distances to all training points) is evaluated online in registers.
3. The forward-solve $L v = k_{cross}$ is executed iteratively.
4. Mean and variance are computed:
   $$\mu = k_{cross}^T \alpha, \quad \sigma^2 = \sigma_{signal}^2 - v^T v$$
5. Standard deviation $\sigma = \sqrt{\sigma^2}$ is evaluated.
6. The analytical Expected Improvement (EI) is calculated and written directly to global memory. No intermediate $m \times n$ covariance matrix is ever allocated.
