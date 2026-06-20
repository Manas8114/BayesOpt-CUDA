#include <cuda_runtime.h>
#include <math.h>
#include <cstdint>

// Fast approximation of standard normal CDF using hardware erff
__device__ __forceinline__ float norm_cdf_fused(float z) {
    const float inv_sqrt_2 = 0.7071067811865475f;
    return 0.5f * (1.0f + erff(z * inv_sqrt_2));
}

// Standard normal PDF
__device__ __forceinline__ float norm_pdf_fused(float x) {
    const float inv_sqrt_2pi = 0.3989422804f;
    return inv_sqrt_2pi * expf(-0.5f * x * x);
}

// Warp-level reduction (single warp)
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

enum class AcquisitionType {
    EI = 0,
    UCB = 1,
    PI = 2
};

// Shared memory layout:
//   [0 .. n-1]         : k_s  (cross-kernel vector)
//   [n .. n + 32 - 1]  : mu reduction scratch (one float per warp)
//   [n+32 .. n+64-1]   : var reduction scratch (one float per warp)
__global__ void acquisition_fused_kernel(
    const float* __restrict__ X_test,
    const float* __restrict__ X_train,
    const float* __restrict__ alpha,
    const float* __restrict__ K_inv,
    float* __restrict__ out,
    const int m,
    const int n,
    const int d,
    const float lengthscale,
    const float variance,
    const float y_best,
    const float exploration_param,
    const AcquisitionType acq_type
) {
    // Shared memory regions
    extern __shared__ float smem[];
    float* k_s       = smem;           // [n]
    float* mu_scratch  = smem + n;     // [32]
    float* var_scratch = smem + n + 32; // [32]

    int test_idx = blockIdx.x;
    if (test_idx >= m) return;

    int lane = threadIdx.x % warpSize;
    int wid  = threadIdx.x / warpSize;
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;

    // -------------------------------------------------------------------------
    // Step 1: fill k_s[i] = k(x_test, x_train_i)
    // -------------------------------------------------------------------------
    const float* x_test = X_test + test_idx * d;

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float dist_sq = 0.0f;
        const float* x_train_i = X_train + i * d;
        for (int j = 0; j < d; ++j) {
            float diff = x_test[j] - x_train_i[j];
            dist_sq += diff * diff;
        }
        k_s[i] = variance * expf(-dist_sq / (2.0f * lengthscale * lengthscale));
    }
    __syncthreads();

    // -------------------------------------------------------------------------
    // Step 2: Compute partial mu and var_red for this thread
    // -------------------------------------------------------------------------
    float mu_local  = 0.0f;
    float var_local = 0.0f;

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float k_i = k_s[i];
        mu_local += k_i * alpha[i];

        // (K_inv @ k_s)[i] = sum_j K_inv[i,j] * k_s[j]
        const float* K_inv_row = K_inv + i * n;
        float row_sum = 0.0f;
        for (int j = 0; j < n; ++j) {
            row_sum += K_inv_row[j] * k_s[j];
        }
        var_local += k_i * row_sum;
    }

    // -------------------------------------------------------------------------
    // Step 3: Block-reduce mu  (two-level: warp then inter-warp)
    // -------------------------------------------------------------------------
    // --- warp level ---
    mu_local = warp_reduce_sum(mu_local);
    if (lane == 0) mu_scratch[wid] = mu_local;
    __syncthreads();

    // --- inter-warp level (only first warp) ---
    float mu_val = (threadIdx.x < num_warps) ? mu_scratch[threadIdx.x] : 0.0f;
    if (wid == 0) {
        mu_val = warp_reduce_sum(mu_val);
        if (lane == 0) mu_scratch[0] = mu_val;  // store final
    }
    __syncthreads();

    // -------------------------------------------------------------------------
    // Step 4: Block-reduce var_red  (same two-level pattern)
    // -------------------------------------------------------------------------
    var_local = warp_reduce_sum(var_local);
    if (lane == 0) var_scratch[wid] = var_local;
    __syncthreads();

    float var_val = (threadIdx.x < num_warps) ? var_scratch[threadIdx.x] : 0.0f;
    if (wid == 0) {
        var_val = warp_reduce_sum(var_val);
        if (lane == 0) var_scratch[0] = var_val;
    }
    __syncthreads();

    // -------------------------------------------------------------------------
    // Step 5: Compute and write acquisition value (thread 0 only)
    // -------------------------------------------------------------------------
    if (threadIdx.x == 0) {
        float mu      = mu_scratch[0];
        float var_red = var_scratch[0];

        float posterior_var = variance - var_red;
        if (posterior_var < 0.0f) posterior_var = 0.0f;
        float posterior_std = sqrtf(posterior_var);

        float val = 0.0f;
        if (acq_type == AcquisitionType::EI) {
            float delta = mu - y_best - exploration_param;
            if (posterior_std > 1e-9f) {
                float z = delta / posterior_std;
                val = delta * norm_cdf_fused(z) + posterior_std * norm_pdf_fused(z);
                if (val < 0.0f) val = 0.0f;
            } else {
                // std ≈ 0: EI = max(0, delta)
                val = delta > 0.0f ? delta : 0.0f;
            }
        } else if (acq_type == AcquisitionType::UCB) {
            val = mu + sqrtf(exploration_param) * posterior_std;
        } else if (acq_type == AcquisitionType::PI) {
            float diff = mu - y_best - exploration_param;
            if (posterior_std > 1e-9f) {
                val = norm_cdf_fused(diff / posterior_std);
            } else {
                val = diff > 0.0f ? 1.0f : 0.0f;
            }
        }
        out[test_idx] = val;
    }
}

// C++ launcher
void launch_acquisition_fused(
    const float* X_test,
    const float* X_train,
    const float* alpha,
    const float* K_inv,
    float* out,
    const int m,
    const int n,
    const int d,
    const float lengthscale,
    const float variance,
    const float y_best,
    const float exploration_param,
    const int acq_type_int
) {
    // Choose block size: round n up to nearest warp, capped at 1024
    int threads = ((n + 31) / 32) * 32;
    if (threads < 32)  threads = 32;
    if (threads > 1024) threads = 1024;

    dim3 blocks(m);
    // Shared: k_s[n] + mu_scratch[32] + var_scratch[32]
    int shared_mem_bytes = (n + 64) * sizeof(float);

    acquisition_fused_kernel<<<blocks, threads, shared_mem_bytes>>>(
        X_test, X_train, alpha, K_inv, out,
        m, n, d, lengthscale, variance, y_best, exploration_param,
        static_cast<AcquisitionType>(acq_type_int)
    );
}
