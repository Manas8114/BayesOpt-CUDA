/**
 * Naive Expected Improvement Kernel
 * 
 * One thread per output element. Elementwise EI computation.
 * 
 * EI(x) = E[max(0, f(x) - y_best - xi)]
 *       = (mean - y_best - xi) * Phi(Z) + std * phi(Z)
 *       where Z = (mean - y_best - xi) / std
 * 
 * For std == 0 (exact interpolation), EI = 0.
 * 
 * Inputs:
 *   mean: (m,) float32 - predictive mean
 *   std:  (m,) float32 - predictive standard deviation
 * Output:
 *   ei:   (m,) float32 - expected improvement values
 * Scalars:
 *   y_best: best observed value so far
 *   xi: exploration parameter
 * 
 * Note: This uses the standard normal CDF/PDF approximations from CUDA math
 * For production, we might want more accurate approximations.
 */

#include <cuda_runtime.h>
#include <math.h>
#include <cstdint>

// Fast approximation of standard normal CDF (Phi)
// Using the Abramowitz & Stegun approximation (7.1.26)
// Phi(z) = 0.5 * [1 + erf(z / sqrt(2))]
__device__ __forceinline__ float phi_cdf(float z) {
    // Use erf for hardware-accelerated CDF
    const float inv_sqrt_2 = 0.7071067811865475f;  // 1/sqrt(2)
    return 0.5f * (1.0f + erff(z * inv_sqrt_2));
}

// Fast approximation of standard normal PDF (phi)
// phi(z) = exp(-z^2 / 2) / sqrt(2*pi)
__device__ __forceinline__ float phi_pdf(float z) {
    const float inv_sqrt_2pi = 0.3989422804014327f;  // 1/sqrt(2*pi)
    return inv_sqrt_2pi * expf(-0.5f * z * z);
}

__global__ void expected_improvement_naive_kernel(
    const float* __restrict__ mean,
    const float* __restrict__ std,
    float* __restrict__ ei,
    const int m,
    const float y_best,
    const float xi
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= m) return;
    
    const float mean_val = mean[idx];
    const float std_val = std[idx];
    
    // Handle std == 0 case (exact interpolation at training points)
    if (std_val <= 1e-12f) {
        ei[idx] = 0.0f;
        return;
    }
    
    const float improvement = mean_val - y_best - xi;
    const float Z = improvement / std_val;
    
    const float Phi = phi_cdf(Z);
    const float phi = phi_pdf(Z);
    
    ei[idx] = improvement * Phi + std_val * phi;
}


void expected_improvement_naive_launch(
    const float* mean,
    const float* std,
    float* ei,
    const int m,
    const float y_best,
    const float xi
) {
    const int block_size = 256;
    const int grid_size = (m + block_size - 1) / block_size;
    
    expected_improvement_naive_kernel<<<grid_size, block_size>>>(
        mean, std, ei, m, y_best, xi
    );
}


__global__ void upper_confidence_bound_naive_kernel(
    const float* __restrict__ mean,
    const float* __restrict__ std,
    float* __restrict__ ucb,
    const int m,
    const float beta
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= m) return;
    
    ucb[idx] = mean[idx] + sqrtf(beta) * std[idx];
}


void upper_confidence_bound_naive_launch(
    const float* mean,
    const float* std,
    float* ucb,
    const int m,
    const float beta
) {
    const int block_size = 256;
    const int grid_size = (m + block_size - 1) / block_size;
    
    upper_confidence_bound_naive_kernel<<<grid_size, block_size>>>(
        mean, std, ucb, m, beta
    );
}


__global__ void probability_of_improvement_naive_kernel(
    const float* __restrict__ mean,
    const float* __restrict__ std,
    float* __restrict__ pi,
    const int m,
    const float y_best,
    const float xi
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= m) return;
    
    const float mean_val = mean[idx];
    const float std_val = std[idx];
    
    if (std_val <= 1e-12f) {
        pi[idx] = 0.0f;
        return;
    }
    
    const float Z = (mean_val - y_best - xi) / std_val;
    pi[idx] = phi_cdf(Z);
}


void probability_of_improvement_naive_launch(
    const float* mean,
    const float* std,
    float* pi,
    const int m,
    const float y_best,
    const float xi
) {
    const int block_size = 256;
    const int grid_size = (m + block_size - 1) / block_size;
    
    probability_of_improvement_naive_kernel<<<grid_size, block_size>>>(
        mean, std, pi, m, y_best, xi
    );
}