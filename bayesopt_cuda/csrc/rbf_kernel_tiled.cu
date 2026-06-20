#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <cstdint>

#define TILE_SIZE 16

__global__ void rbf_kernel_tiled_kernel(
    const float* __restrict__ X,
    const float* __restrict__ Y,
    float* __restrict__ K,
    const int n,
    const int m,
    const int d,
    const float lengthscale,
    const float variance
) {
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float dist_sq = 0.0f;

    __shared__ float X_s[TILE_SIZE][TILE_SIZE];
    __shared__ float Y_s[TILE_SIZE][TILE_SIZE];

    for (int k = 0; k < (d + TILE_SIZE - 1) / TILE_SIZE; ++k) {
        // Load X tile
        int x_row = blockIdx.y * TILE_SIZE + threadIdx.y;
        int x_col = k * TILE_SIZE + threadIdx.x;
        if (x_row < n && x_col < d) {
            X_s[threadIdx.y][threadIdx.x] = X[x_row * d + x_col];
        } else {
            X_s[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load Y tile
        int y_row = blockIdx.x * TILE_SIZE + threadIdx.y;
        int y_col = k * TILE_SIZE + threadIdx.x;
        if (y_row < m && y_col < d) {
            Y_s[threadIdx.y][threadIdx.x] = Y[y_row * d + y_col];
        } else {
            Y_s[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial distance
        for (int i = 0; i < TILE_SIZE; ++i) {
            if (k * TILE_SIZE + i < d) {
                float diff = X_s[threadIdx.y][i] - Y_s[threadIdx.x][i];
                dist_sq += diff * diff;
            }
        }
        
        __syncthreads();
    }

    if (row < n && col < m) {
        float arg = -dist_sq / (2.0f * lengthscale * lengthscale);
        K[row * m + col] = variance * expf(arg);
    }
}

void launch_rbf_kernel_tiled(
    const float* X,
    const float* Y,
    float* K,
    const int n,
    const int m,
    const int d,
    const float lengthscale,
    const float variance
) {
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((m + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);
    
    rbf_kernel_tiled_kernel<<<blocks, threads>>>(X, Y, K, n, m, d, lengthscale, variance);
}
