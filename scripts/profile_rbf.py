import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import bayesopt_cuda._C as _C
from bayesopt_cuda.kernels_optimized import rbf_kernel_tiled

def main():
    n, m, d = 2048, 2048, 32
    X = torch.randn(n, d, dtype=torch.float32, device="cuda")
    Y = torch.randn(m, d, dtype=torch.float32, device="cuda")
    ls, var = 1.0, 1.0
    
    # Warmup
    for _ in range(3):
        rbf_kernel_tiled(X, Y, ls, var)
    torch.cuda.synchronize()
    
    # Profile this call
    rbf_kernel_tiled(X, Y, ls, var)
    torch.cuda.synchronize()

if __name__ == "__main__":
    main()
