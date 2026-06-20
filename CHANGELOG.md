# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-06-20
### Added
- **O(n) Cholesky Acquisition Fused Kernel**: An optimised implementation that computes posterior variance analytically by solving the lower-triangular Cholesky factor $L$ ($Lv = k_s$) iteratively in-kernel. This avoids $O(n^2)$ shared memory spills and allows scaling to thousands of training points.
- **Mixed-Precision FP16 RBF Kernel**: Utilises `__half2` SIMD types, explicitly loading 16-bit floats from global memory into registers, which halves DRAM bandwidth pressure and yields significant speedups over PyTorch native implementations.
- **Tiled RBF Kernel**: Implemented $16 \times 16$ 2D shared-memory tiling to amortise global memory traffic.
- **Python Wrapper**: Integrated all kernels via a `scikit-learn`/`BoTorch` compatible interface.
- **Benchmarks**: Benchmarking suite via `benchmarks/bench_kernels.py` utilising hardware CUDA events.
- **Interactive Demo**: Added `demo.py` to showcase end-to-end GP training and EI evaluation in real time.
- **Packaging Tools**: Added `pyproject.toml` and `Dockerfile` to provide robust, cross-platform PyPI builds.
