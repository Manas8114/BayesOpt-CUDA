# BayesOpt-CUDA 🚀

[![CUDA](https://img.shields.io/badge/CUDA-12.1%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A high-performance, GPU-accelerated Bayesian Optimization library. It features custom, hand-optimized CUDA kernels for covariance matrix construction and fused acquisition function evaluation, exposed via a `scikit-learn` and `BoTorch`-compatible Python wrapper.

---

## 📖 Table of Contents
* [Key Features](#-key-features)
* [System Architecture](#%EF%B8%8F-system-architecture)
* [Detailed Documentation](#-detailed-documentation)
* [Installation](#-installation)
  * [Prerequisites](#prerequisites)
  * [From Source](#from-source)
  * [Docker Build](#docker-build)
* [Quick Start](#-quick-start)
  * [1. Gaussian Process Regression](#1-gaussian-process-regression)
  * [2. Bayesian Optimization Loop](#2-bayesian-optimization-loop)
  * [3. Running the Interactive Demo](#3-running-the-interactive-demo)
* [Performance Benchmarks](#-performance-benchmarks)
* [Testing](#-testing)
* [License](#-license)

---

## ⚡ Key Features

* **Tiled RBF Kernel**: Computes the Gaussian Radial Basis Function (RBF) covariance matrix using a custom $16 \times 16$ 2D shared-memory tiling layout. Amortises global memory bandwidth, yielding up to **3.26x speedup** over naive implementations.
* **Mixed-Precision FP16 RBF**: Utilises native CUDA SIMD `__half2` arithmetic to half DRAM bandwidth pressure. Performs accumulation and exponentiation in float32 to preserve high numeric precision and range, achieving up to **2.21x speedup** at intermediate dimensions.
* **Fused Acquisition Functions**: Fuses Gaussian Process posterior mean and variance computations with Expected Improvement (EI), Upper Confidence Bound (UCB), and Probability of Improvement (PI) into a single unified kernel pass. Completely avoids materialising intermediate $m \times n$ covariance matrices in global memory.
* **$O(n)$ Cholesky Variance Solve**: Computes the predictive variance directly from the lower-triangular Cholesky factor $L$ via parallelized, fused forward-substitution inside the kernel. Restricts shared memory footprint to $O(n)$ instead of $O(n^2)$, allowing the kernel to scale to thousands of training points without spilling or crashing.

---

## 🏗️ System Architecture

The package bridges high-level Python wrappers with low-level CUDA threads through a PyBind11 bindings module. The inner Cholesky decomposition of the covariance matrix ($K_{train} = L L^T$) is computed on the GPU using PyTorch's native `linalg` wrappers, while kernel distances and acquisition evaluations are executed in custom CUDA blocks.

```mermaid
graph TD
    subgraph Python Host API
        BO[BayesianOptimizerCUDA] -->|"Runs loop"| GP[GaussianProcessRegressorCUDA]
        GP -->|"Wrapper calls"| KO[bayesopt_cuda.kernels_optimized]
    end

    subgraph C++ Bindings (PyBind11)
        KO -->|"Dispatches tensors"| PB[bayesopt_cuda._C bindings.cpp]
    end

    subgraph Custom CUDA Kernels
        PB -->|"Launch"| RK_tiled[rbf_kernel_tiled.cu]
        PB -->|"Launch"| RK_fp16[rbf_kernel_tiled_fp16.cu]
        PB -->|"Launch"| ACQ_chol[acquisition_fused_cholesky.cu]
    end

    subgraph GPU Hardware
        RK_fp16 -->|"SIMD __half2 & DRAM reduction"| HW_FP16[RTX Tensor Cores / ALU]
        ACQ_chol -->|"O(n) SMEM Forward-Substitution"| HW_SMEM[Shared Memory L1 Cache]
    end
```

---

## 📚 Detailed Documentation

For deep technical insights, equations, and API parameters, consult the following documentation modules:

* [Architecture & Data Flows](file:///c:/Users/msgok/OneDrive/Desktop/Project/Hermes/bayesopt_cuda/docs/architecture.md): Visualises the execution pipelines, JIT fallback compilation, and data transformations.
* [API Reference](file:///c:/Users/msgok/OneDrive/Desktop/Project/Hermes/bayesopt_cuda/docs/api.md): Detailed parameter lists, method signatures, and return values for all Python classes and C++ bindings.
* [Kernel & Math Design](file:///c:/Users/msgok/OneDrive/Desktop/Project/Hermes/bayesopt_cuda/docs/kernels.md): The mathematical formulations of RBF covariance and acquisition types (EI, UCB, PI), alongside shared-memory tiling layouts.
* [Performance Benchmarks](file:///c:/Users/msgok/OneDrive/Desktop/Project/Hermes/bayesopt_cuda/docs/benchmarks.md): Hardware setup details, comparative tables, and roofline/scaling limits analyses.

---

## ⚙️ Installation

### Prerequisites
* **OS**: Windows 10/11 or Linux
* **Python**: 3.11+
* **PyTorch**: 2.0+ (compiled with CUDA support)
* **CUDA Toolkit**: 12.1+ (with matching `nvcc` compiler)
* **C++ Compiler**: MSVC 2022 (Windows) or GCC 9+ (Linux)

### From Source
1. Clone the repository:
   ```bash
   git clone https://github.com/example/bayesopt_cuda.git
   cd bayesopt_cuda
   ```
2. Build and install the package in editable mode:
   ```bash
   pip install -e .
   ```

*(On Windows, if MSVC compiler paths are not registered in your environment, run [build.bat](file:///c:/Users/msgok/OneDrive/Desktop/Project/Hermes/bayesopt_cuda/build.bat) to automatically load `vcvarsall.bat` and compile the package).*

### Docker Build
A multi-stage Dockerfile is provided to construct a hermetic Ubuntu environment with Python 3.11, CUDA 12.1, and PyTorch pre-installed:
```bash
docker build -t bayesopt-cuda .
docker run --gpus all -it bayesopt-cuda
```

---

## 🚀 Quick Start

### 1. Gaussian Process Regression
```python
import numpy as np
import torch
from bayesopt_cuda.gp_cuda import GaussianProcessRegressorCUDA

# Create random training data
X_train = np.random.uniform(-3.0, 3.0, size=(100, 2)).astype(np.float32)
y_train = np.sin(X_train[:, 0]) * np.cos(X_train[:, 1])

# Initialise and fit the GPU GP Regressor
gp = GaussianProcessRegressorCUDA(
    lengthscale=1.0, 
    variance=1.0, 
    noise=1e-5, 
    device="cuda"
)
gp.fit(X_train, y_train)

# Predict at test points
X_test = np.random.uniform(-3.0, 3.0, size=(1000, 2)).astype(np.float32)
mean, std = gp.predict(X_test, return_std=True)

print(f"Mean shape: {mean.shape}, Std shape: {std.shape}")
```

### 2. Bayesian Optimization Loop
```python
import numpy as np
from bayesopt_cuda.optimizer_cuda import BayesianOptimizerCUDA
from bayesopt_cuda.reference import branin

# Objective function to maximize (Branin function, negated for maximization)
def objective(x):
    return float(-branin(x))

# Define parameter bounds
bounds = np.array([[-5.0, 10.0], [0.0, 15.0]], dtype=np.float32)

# Initialise GPU Bayesian Optimizer
opt = BayesianOptimizerCUDA(
    objective=objective,
    bounds=bounds,
    acquisition="ei",
    n_candidates=5000,
    n_local_restarts=10,
    device="cuda"
)

# Run for 20 iterations
X_history, y_history = opt.run(n_iter=20)
print(f"Optimal parameter found: {opt.best_x}")
print(f"Maximum objective value: {opt.best_y}")
```

### 3. Running the Interactive Demo
Test the installation and benchmark your GPU against a synthetic GP optimization benchmark:
```bash
python demo.py --n 1024 --m 4096 --d 16
```

---

## 📊 Performance Benchmarks

The benchmarks below were collected on an **NVIDIA GeForce RTX 5080 Laptop GPU** (60 SMs, Blackwell Architecture, CUDA 13.3) and PyTorch 2.6.0.

### RBF Kernel Execution Time
*Comparison of naive PyTorch, our custom tiled RBF (FP32), and mixed-precision (FP16) kernels.*

| Dataset Size (n, m, d) | Naive PyTorch (ms) | Tiled FP32 (ms) | Tiled FP16 (ms) | Speedup (FP32 vs Naive) | Speedup (FP16 vs Naive) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **(512, 512, 8)** | 0.057 | 0.023 | 0.021 | 2.53x | 2.73x |
| **(1024, 1024, 16)** | 0.084 | 0.054 | 0.038 | 1.53x | **2.21x** |
| **(2048, 2048, 32)** | 0.939 | 0.288 | 0.311 | **3.26x** | **3.02x** |

### Fused Expected Improvement (EI) Evaluation
*Comparison of naive PyTorch composition vs our fused shared-memory kernels.*

| Dataset Size (n, m, d) | Naive PyTorch (ms) | Fused $K^{-1}$ (ms) | Fused Cholesky (ms) | Speedup ($K^{-1}$) | Speedup (Cholesky) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **(32, 256, 2)** | 0.148 | 0.031 | 0.036 | **4.83x** | **4.14x** |
| **(64, 512, 4)** | 0.115 | 0.042 | 0.060 | **2.75x** | **1.93x** |

> [!NOTE]
> For small scales ($n \le 64$), the custom fused kernels deliver a massive **~4.8x speedup** by avoiding DRAM allocations and roundtrips.
>
> For large scales ($n \ge 128$), PyTorch's native execution becomes faster. This is because PyTorch routes large matrix operations through cuBLAS (Tensor Cores), which provide far greater raw FLOPS throughput than custom CUDA core instructions can achieve.

---

## 🧪 Testing

Correctness tests ensure custom kernels are numerically verified to match SciPy and PyTorch reference implementations. Run tests via `pytest`:

```bash
pip install pytest
python -m pytest tests/ -v 
```

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
#   B a y e s O p t - C U D A 
 
 
