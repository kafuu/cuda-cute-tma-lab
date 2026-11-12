#include "ccg/basic_gemm.hpp"

namespace {

__global__
void gemm_naive_kernel(
    const float *a,
    const float *b,
    float *c,
    int m,
    int n,
    int k)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= m || col >= n) {
        return;
    }

    float sum = 0.0f;
    for (int kk = 0; kk < k; ++kk) {
        sum += a[row * k + kk] * b[kk * n + col];
    }

    c[row * n + col] = sum;
}

}  // namespace

namespace ccg {

cudaError_t launch_naive_gemm_f32(
    const float *a,
    const float *b,
    float *c,
    int m,
    int n,
    int k)
{
    if (a == nullptr || b == nullptr || c == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (m <= 0 || n <= 0 || k <= 0) {
        return cudaErrorInvalidValue;
    }

    constexpr int block_x = 32;
    constexpr int block_y = 8;
    dim3 block(block_x, block_y);
    dim3 grid(
        (n + block_x - 1) / block_x,
        (m + block_y - 1) / block_y);

    gemm_naive_kernel<<<grid, block>>>(a, b, c, m, n, k);
    return cudaGetLastError();
}

}  // namespace ccg
