"""BayesOpt-CUDA: GPU-accelerated Bayesian Optimization with custom CUDA kernels.

The compiled CUDA extension is loaded lazily via bayesopt_cuda._C; importing
this package does not require nvcc or a C++ host compiler. Call
``bayesopt_cuda.load_extension()`` to build/load the GPU kernels.
"""
__version__ = "0.1.0"
from .reference import (
    BayesianOptimizer,
    GaussianProcessRegressor,
    branin,
    hartmann6,
    rbf_kernel,
    expected_improvement,
    upper_confidence_bound,
    probability_of_improvement,
)
from .gp_cuda import GaussianProcessRegressorCUDA
from .optimizer_cuda import BayesianOptimizerCUDA

def load_extension():
    """Build or load the bayesopt_cuda._C extension.

    Tries "pip install" produced .pyd first, falls back to JIT compile
    via torch.utils.cpp_extension.load (cached under ~/.cache/torch_extensions).
    """
    import importlib
    try:
        return importlib.import_module("bayesopt_cuda._C")
    except ImportError:
        pass

    import os
    here = os.path.dirname(os.path.abspath(__file__))
    cs = os.path.join(here, "csrc")
    rbf = os.path.join(cs, "rbf_kernel_naive.cu")
    ei = os.path.join(cs, "ei_kernel_naive.cu")
    bp = os.path.join(cs, "bindings.cpp")
    rbf_tiled = os.path.join(cs, "rbf_kernel_tiled.cu")
    acq_fused = os.path.join(cs, "acquisition_fused.cu")
    acq_fused_chol = os.path.join(cs, "acquisition_fused_cholesky.cu")
    rbf_tiled_fp16 = os.path.join(cs, "rbf_kernel_tiled_fp16.cu")

    from torch.utils.cpp_extension import load
    return load(
        name="bayesopt_cuda_C",
        sources=[bp, rbf, ei, rbf_tiled, acq_fused, acq_fused_chol, rbf_tiled_fp16],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17"],
        verbose=False,
    )

__all__ = [
    # Reference (NumPy/SciPy)
    "BayesianOptimizer", "GaussianProcessRegressor",
    "branin", "hartmann6", "rbf_kernel",
    "expected_improvement", "upper_confidence_bound", "probability_of_improvement",
    # CUDA-accelerated
    "GaussianProcessRegressorCUDA", "BayesianOptimizerCUDA",
    "load_extension", "__version__",
]
