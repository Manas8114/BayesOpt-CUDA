/**
 * rbf_kernel_tiled_fp16.cu
 * =========================
 * Tiled RBF kernel with FP16 input loading and mixed-precision
 * distance accumulation.
 *
 * Strategy:
 *   - Accept X and Y as float16 (__half) tensors (2× smaller than float32
 *     in global memory → 2× better cache utilisation on Blackwell).
 *   - Compute diff in FP16 with __hsub; square with __hmul.
 *   - Accumulate dist_sq in float32 (convert each squared diff from FP16
 *     before adding) — avoids FP16 overflow for large d or large diff values.
 *   - Final expf and output are float32.
 *
 * Tiling: 16×16 block of output elements, with X/Y tiles loaded into shared
 * memory as float16 (TILE × d halves each = TILE*d*2 bytes vs 4 bytes for FP32).
 *
 * On Blackwell (sm_12.0):
 *   - FP16 arithmetic: ~2× the CUDA-core throughput of FP32 arithmetic.
 *   - The bottleneck for large n/d is memory bandwidth; FP16 inputs reduce
 *     effective bandwidth demand by 2×.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

#define TILE 16

/**
 * Shared memory layout (per block):
 *   X_tile: TILE * d halves  (loaded as __half2 for 2-wide SIMD)
 *   Y_tile: TILE * d halves
 *
 * But d is runtime variable, so we use dynamic shared mem:
 *   smem[0..TILE*d-1]          : X_tile (__half)
 *   smem[TILE*d..2*TILE*d-1]   : Y_tile (__half)
 */
__global__ void rbf_kernel_tiled_fp16_kernel(
    const __half* __restrict__ X,
    const __half* __restrict__ Y,
    float*        __restrict__ K,
    const int n, const int m, const int d,
    const float   inv_ls2_half,   // -1 / (2 * ls^2), passed pre-computed
    const float   variance
) {
    extern __shared__ __half smem_h[];
    __half* X_tile = smem_h;           // [TILE * d]
    __half* Y_tile = smem_h + TILE * d; // [TILE * d]

    const int row = blockIdx.x * TILE + threadIdx.x; // output row (n dim)
    const int col = blockIdx.y * TILE + threadIdx.y; // output col (m dim)

    // Load X_tile row threadIdx.x from X[row, :]
    if (row < n) {
        for (int k = threadIdx.y; k < d; k += TILE)
            X_tile[threadIdx.x * d + k] = X[row * d + k];
    }
    // Load Y_tile row threadIdx.y from Y[col, :]
    if (col < m) {
        for (int k = threadIdx.x; k < d; k += TILE)
            Y_tile[threadIdx.y * d + k] = Y[col * d + k];
    }
    __syncthreads();

    if (row >= n || col >= m) return;

    // Compute squared Euclidean distance in mixed precision
    float dist_sq = 0.0f;
    const __half* xi = X_tile + threadIdx.x * d;
    const __half* yj = Y_tile + threadIdx.y * d;

    // Process pairs of dimensions with __half2 for SIMD throughput
    int k = 0;
    for (; k + 1 < d; k += 2) {
        __half2 xi2 = __halves2half2(xi[k], xi[k+1]);
        __half2 yj2 = __halves2half2(yj[k], yj[k+1]);
        __half2 diff2 = __hsub2(xi2, yj2);
        __half2 sq2   = __hmul2(diff2, diff2);
        // Convert each FP16 square to FP32 before accumulating (avoids overflow)
        dist_sq += __half2float(sq2.x) + __half2float(sq2.y);
    }
    // Handle odd trailing dimension
    if (k < d) {
        __half diff = __hsub(xi[k], yj[k]);
        dist_sq += __half2float(__hmul(diff, diff));
    }

    K[row * m + col] = variance * expf(dist_sq * inv_ls2_half);
}

// C++ launcher
void launch_rbf_kernel_tiled_fp16(
    const __half* X, const __half* Y, float* K,
    const int n, const int m, const int d,
    const float lengthscale, const float variance
) {
    float inv_ls2_half = -1.0f / (2.0f * lengthscale * lengthscale);

    dim3 threads(TILE, TILE);
    dim3 blocks((n + TILE - 1) / TILE, (m + TILE - 1) / TILE);

    // Dynamic shared memory: X_tile (TILE*d halves) + Y_tile (TILE*d halves)
    size_t smem_bytes = 2 * TILE * d * sizeof(__half);

    rbf_kernel_tiled_fp16_kernel<<<blocks, threads, smem_bytes>>>(
        X, Y, K,
        n, m, d,
        inv_ls2_half, variance
    );
}
