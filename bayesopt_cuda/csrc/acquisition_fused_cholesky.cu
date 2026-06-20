/**
 * acquisition_fused_cholesky.cu — revised
 * ========================================
 * Fused EI/UCB/PI using Cholesky factor L for posterior variance.
 * O(n) shared memory instead of O(n²) K_inv.
 *
 * Bug fix: row-solve scratch must not alias mu_scratch (which holds the
 * final mu value). Use a dedicated third scratch region "row_scratch[32]".
 *
 * Shared memory layout:
 *   [0 .. n-1]            : k_s       (cross-kernel vector)
 *   [n .. 2n-1]           : v         (Cholesky solve: Lv = k_s)
 *   [2n .. 2n+31]         : mu_scratch [32]
 *   [2n+32 .. 2n+63]      : var_scratch [32]
 *   [2n+64 .. 2n+95]      : row_scratch [32]  (inner-product per row-solve step)
 * Total SMEM: (2n + 96) * sizeof(float)
 *   At n=512:  4.37 KB  vs. 1026 KB for K_inv approach.
 */

#include <cuda_runtime.h>
#include <math.h>
#include <cstdint>

__device__ __forceinline__ float norm_cdf_chol(float z) {
    const float inv_sqrt_2 = 0.7071067811865475f;
    return 0.5f * (1.0f + erff(z * inv_sqrt_2));
}
__device__ __forceinline__ float norm_pdf_chol(float x) {
    const float inv_sqrt_2pi = 0.3989422804f;
    return inv_sqrt_2pi * expf(-0.5f * x * x);
}
__device__ __forceinline__ float warp_reduce_chol(float val) {
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Two-level block reduction: puts result in scratch[0], returns it for thread 0.
__device__ __forceinline__ float block_reduce_to_scratch(
        float val, float* scratch, int num_warps) {
    int lane = threadIdx.x % warpSize;
    int wid  = threadIdx.x / warpSize;
    val = warp_reduce_chol(val);
    if (lane == 0) scratch[wid] = val;
    __syncthreads();
    float out = (threadIdx.x < num_warps) ? scratch[threadIdx.x] : 0.0f;
    if (wid == 0) {
        out = warp_reduce_chol(out);
        if (lane == 0) scratch[0] = out;
    }
    __syncthreads();
    return out;
}

enum class AcqTypeChol { EI = 0, UCB = 1, PI = 2 };

__global__ void acquisition_fused_chol_kernel(
    const float* __restrict__ X_test,
    const float* __restrict__ X_train,
    const float* __restrict__ alpha,
    const float* __restrict__ L,
    float* __restrict__ out,
    const int m, const int n, const int d,
    const float lengthscale, const float variance,
    const float y_best, const float exploration_param,
    const AcqTypeChol acq_type
) {
    extern __shared__ float smem[];
    float* k_s         = smem;           // [n]
    float* v           = smem + n;       // [n]
    float* mu_scratch  = smem + 2*n;     // [32]
    float* var_scratch = smem + 2*n + 32;// [32]
    float* row_scratch = smem + 2*n + 64;// [32]  — dedicated to row-solve

    const int test_idx = blockIdx.x;
    if (test_idx >= m) return;

    const int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    const int lane      = threadIdx.x % warpSize;
    const int wid       = threadIdx.x / warpSize;

    // -------------------------------------------------------------------------
    // Step 1: Fill k_s[i] = variance * exp(-||x_test - x_train_i||² / 2ls²)
    // -------------------------------------------------------------------------
    const float* x_test = X_test + test_idx * d;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float dist_sq = 0.0f;
        const float* xt = X_train + i * d;
        for (int j = 0; j < d; ++j) {
            float diff = x_test[j] - xt[j];
            dist_sq += diff * diff;
        }
        k_s[i] = variance * expf(-dist_sq / (2.0f * lengthscale * lengthscale));
        v[i] = 0.0f;  // zero-init v here to avoid a second pass
    }
    __syncthreads();

    // -------------------------------------------------------------------------
    // Step 2: mu = k_s @ alpha  (parallel block reduction)
    // -------------------------------------------------------------------------
    float mu_local = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        mu_local += k_s[i] * alpha[i];

    block_reduce_to_scratch(mu_local, mu_scratch, num_warps);
    // mu_scratch[0] now holds mu — do NOT touch mu_scratch until step 5.

    // -------------------------------------------------------------------------
    // Step 3: Forward substitution Lv = k_s   (row-by-row, v in SMEM)
    //
    // v[i] = (k_s[i] - sum_{j<i} L[i,j]*v[j]) / L[i,i]
    //
    // All blockDim.x threads cooperate on each row's inner product,
    // synchronised via row_scratch.  Sequential inter-row dependency handled
    // by __syncthreads() after each row.
    // -------------------------------------------------------------------------
    for (int i = 0; i < n; ++i) {
        float inner_local = 0.0f;
        const float* L_row = L + i * n;
        for (int j = threadIdx.x; j < i; j += blockDim.x)
            inner_local += L_row[j] * v[j];

        // Reduce inner_local into row_scratch[0]
        inner_local = warp_reduce_chol(inner_local);
        if (lane == 0) row_scratch[wid] = inner_local;
        __syncthreads();

        if (wid == 0) {
            float t = (threadIdx.x < num_warps) ? row_scratch[threadIdx.x] : 0.0f;
            t = warp_reduce_chol(t);
            if (lane == 0) {
                float diag = L_row[i];
                v[i] = (fabsf(diag) > 1e-10f) ? (k_s[i] - t) / diag : 0.0f;
            }
        }
        __syncthreads();  // v[i] visible for next row
    }

    // -------------------------------------------------------------------------
    // Step 4: var_red = ||v||²  (parallel block reduction)
    // -------------------------------------------------------------------------
    float var_local = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        var_local += v[i] * v[i];

    block_reduce_to_scratch(var_local, var_scratch, num_warps);
    // var_scratch[0] now holds var_red.

    // -------------------------------------------------------------------------
    // Step 5: Compute and write acquisition value (thread 0)
    // -------------------------------------------------------------------------
    if (threadIdx.x == 0) {
        float mu      = mu_scratch[0];
        float var_red = var_scratch[0];

        float posterior_var = variance - var_red;
        if (posterior_var < 0.0f) posterior_var = 0.0f;
        float posterior_std = sqrtf(posterior_var);

        float val = 0.0f;
        if (acq_type == AcqTypeChol::EI) {
            float delta = mu - y_best - exploration_param;
            if (posterior_std > 1e-9f) {
                float z = delta / posterior_std;
                val = delta * norm_cdf_chol(z) + posterior_std * norm_pdf_chol(z);
                if (val < 0.0f) val = 0.0f;
            } else {
                val = delta > 0.0f ? delta : 0.0f;
            }
        } else if (acq_type == AcqTypeChol::UCB) {
            val = mu + sqrtf(exploration_param) * posterior_std;
        } else if (acq_type == AcqTypeChol::PI) {
            float diff = mu - y_best - exploration_param;
            if (posterior_std > 1e-9f) {
                val = norm_cdf_chol(diff / posterior_std);
            } else {
                val = diff > 0.0f ? 1.0f : 0.0f;
            }
        }
        out[test_idx] = val;
    }
}

void launch_acquisition_fused_chol(
    const float* X_test, const float* X_train,
    const float* alpha, const float* L,
    float* out, const int m, const int n, const int d,
    const float lengthscale, const float variance,
    const float y_best, const float exploration_param,
    const int acq_type_int
) {
    int threads = ((n + 31) / 32) * 32;
    if (threads < 32)  threads = 32;
    if (threads > 256) threads = 256;

    dim3 blocks(m);
    // k_s[n] + v[n] + mu_scratch[32] + var_scratch[32] + row_scratch[32]
    int shared_mem_bytes = (2 * n + 96) * sizeof(float);

    acquisition_fused_chol_kernel<<<blocks, threads, shared_mem_bytes>>>(
        X_test, X_train, alpha, L, out,
        m, n, d, lengthscale, variance, y_best, exploration_param,
        static_cast<AcqTypeChol>(acq_type_int)
    );
}
