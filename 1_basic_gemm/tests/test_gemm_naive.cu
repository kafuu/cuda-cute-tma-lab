#include "ccg/basic_gemm.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

float input_value(int index, int multiplier)
{
    return static_cast<float>((index * multiplier) % 19 - 9) * 0.125f;
}

}  // namespace

int main()
{
    constexpr int m = 17;
    constexpr int n = 19;
    constexpr int k = 23;

    std::vector<float> a(m * k);
    std::vector<float> b(k * n);
    std::vector<float> c(m * n, 0.0f);
    std::vector<float> reference(m * n, 0.0f);

    for (int i = 0; i < m * k; ++i) {
        a[i] = input_value(i, 3);
    }
    for (int i = 0; i < k * n; ++i) {
        b[i] = input_value(i, 5);
    }

    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            float sum = 0.0f;
            for (int kk = 0; kk < k; ++kk) {
                sum += a[row * k + kk] * b[kk * n + col];
            }
            reference[row * n + col] = sum;
        }
    }

    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;

    cudaMalloc(&d_a, a.size() * sizeof(float));
    cudaMalloc(&d_b, b.size() * sizeof(float));
    cudaMalloc(&d_c, c.size() * sizeof(float));

    cudaMemcpy(d_a, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice);

    cudaError_t error = ccg::launch_naive_gemm_f32(d_a, d_b, d_c, m, n, k);
    if (error == cudaSuccess) {
        error = cudaDeviceSynchronize();
    }
    if (error != cudaSuccess) {
        std::fprintf(stderr, "naive GEMM failed: %s\n", cudaGetErrorString(error));
        return 1;
    }

    cudaMemcpy(c.data(), d_c, c.size() * sizeof(float), cudaMemcpyDeviceToHost);

    float max_error = 0.0f;
    for (int i = 0; i < m * n; ++i) {
        max_error = std::fmax(max_error, std::fabs(c[i] - reference[i]));
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    std::printf("naive GEMM max error: %.8f\n", max_error);
    return max_error <= 1.0e-5f ? 0 : 1;
}
