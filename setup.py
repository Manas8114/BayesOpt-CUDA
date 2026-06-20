"""
BayesOpt-CUDA build script.

Uses torch.utils.cpp_extension's BuildExtension so that
``pip install .`` (or ``pip install -e .``) compiles ``.cu`` files
via the locally installed nvcc using PyTorch's matching CUDA arch list.
"""

import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Filter CUDA arch flags to match PyTorch's supported list.
# This avoids building for architectures the local PyTorch wheel
# wasn't compiled for and keeps the build hermetic.
def torch_cuda_arch_args():
    try:
        import torch
        if not torch.cuda.is_available():
            return []
        cap = torch.cuda.get_device_capability(0)
        arch = f"{cap[0]}{cap[1]}"
        return [f"-gencode=arch=compute_{arch},code=sm_{arch}"]
    except Exception:
        return []

setup(
    ext_modules=[
        CUDAExtension(
            name="bayesopt_cuda._C",
            sources=[
                "bayesopt_cuda/csrc/bindings.cpp",
                "bayesopt_cuda/csrc/rbf_kernel_naive.cu",
                "bayesopt_cuda/csrc/ei_kernel_naive.cu",
                "bayesopt_cuda/csrc/rbf_kernel_tiled.cu",
                "bayesopt_cuda/csrc/acquisition_fused.cu",
                "bayesopt_cuda/csrc/acquisition_fused_cholesky.cu",
                "bayesopt_cuda/csrc/rbf_kernel_tiled_fp16.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-std=c++17",
                    "--expt-relaxed-constexpr",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)
