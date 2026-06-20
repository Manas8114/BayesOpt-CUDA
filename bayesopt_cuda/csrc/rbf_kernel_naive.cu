/**
 * Naive RBF Kernel Matrix Construction
 * 
 * One thread per output element. No shared memory, no tiling.
 * This is the correctness baseline - the optimized version must match this exactly.
 * 
 * k(x, y) = variance * exp(-||x - y||^2 / (2 * lengthscale^2))
 * 
 * Inputs:
 *   X: (n, d) row-major, float32
 *   Y: (m, d) row-major, float32
 * Output:
 *   K: (n, m) row-major, float32
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <cstdint>

__global__ void rbf_kernel_naive_kernel(
    const float* __restrict__ X,
    const float* __restrict__ Y,
    float* __restrict__ K,
    const int n,
    const int m,
    const int d,
    const float lengthscale,
    const float variance
) {
    // One thread per output element (i, j)
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n * m;
    
    if (idx >= total) return;
    
    const int i = idx / m;  // row in X
    const int j = idx % m;  // row in Y
    
    // Compute squared Euclidean distance between X[i] and Y[j]
    float sqdist = 0.0f;
    for (int k = 0; k < d; ++k) {
        const float xi = X[i * d + k];
        const float yj = Y[j * d + k];
        const float diff = xi - yj;
        sqdist += diff * diff;
    }
    
    // Numerical stability: clip negative values from floating point error
    sqdist = fmaxf(sqdist, 0.0f);
    
    // RBF kernel
    const float inv_lengthscale2 = 1.0f / (lengthscale * lengthscale);
    K[idx] = variance * expf(-0.5f * sqdist * inv_lengthscale2);
}


void rbf_kernel_naive_launch(
    const float* X,
    const float* Y,
    float* K,
    const int n,
    const int m,
    const int d,
    const float lengthscale,
    const float variance
) {
    const int total = n * m;
    const int block_size = 256;
    const int grid_size = (total + block_size - 1) / block_size;
    
    rbf_kernel_naive_kernel<<<grid_size, block_size>>>(
        X, Y, K, n, m, d, lengthscale, variance
    );
    
    // Note: caller is responsible for cudaDeviceSynchronize() if needed
}


__global__ void rbf_kernel_naive_batch_kernel(
    const float* __restrict__ X,    // (batch, n, d)
    const float* __restrict__ Y,    // (batch, m, d)
    float* __restrict__ K,          // (batch, n, m)
    const int batch,
    const int n,
    const int m,
    const int d,
    const float lengthscale,
    const float variance
) {
    // One thread per output element (b, i, j)
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch * n * m;
    
    if (idx >= total) return;
    
    const int b = idx / (n * m);
    const int ij = idx % (n * m);
    const int i = ij / m;
    const int j = ij % m;
    
    const float* X_b = X + b * n * d;
    const float* Y_b = Y + b * m * d;
    float* K_b = K + b * n * m;
    
    float sqdist = 0.0f;
    for (int k = 0; k < d; ++k) {
        const float xi = X_b[i * d + k];
        const float yj = Y_b[j * d + k];
        const float diff = xi - yj;
        sqdist += diff * diff;
    }
    
    sqdist = fmaxf(sqdist, 0.0f);
    const float inv_lengthscale2 = 1.0f / (lengthscale * lengthscale);
    K_b[i * m + j] = variance * expf(-0.5f * sqdist * inv_lengthscale2);
}


void rbf_kernel_naive_batch_launch(
    const float* X,
    const float* Y,
    float* K,
    const int batch,
    const int n,
    const int m,
    const int d,
    const float lengthscale,
    const float variance
) {
    const int total = batch * n * m;
    const int block_size = 256;
    const int grid_size = (total + block_size - 1) / block_size;
    
    rbf_kernel_naive_batch_kernel<<<grid_size, block_size>>>(
        X, Y, K, batch, n, m, d, lengthscale, variance
    );
}