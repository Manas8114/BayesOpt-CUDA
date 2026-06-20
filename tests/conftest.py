"""
Pytest configuration for BayesOpt-CUDA.

Tests that exercise the actual CUDA extension are auto-marked with
``cuda_extension``. Use ``pytest -m 'not cuda_extension'`` to run only
CPU-only tests.
"""

import pytest


def pytest_collection_modifyitems(config, items):
    for item in items:
        # Auto-mark anything under test_kernels_*.py as exercising the
        # CUDA extension, since they use torch + bayesopt_cuda._C.
        path_parts = item.location[0].split("\\") if item.location else []
        fname = path_parts[-1] if path_parts else ""
        if fname.startswith("test_kernels_") or fname.startswith("test_optimizer_"):
            item.add_marker(pytest.mark.cuda_extension)
