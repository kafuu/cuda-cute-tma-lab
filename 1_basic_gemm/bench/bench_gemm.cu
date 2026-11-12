#include "ccg/basic_gemm.hpp"

#include <cstdio>

int main()
{
    constexpr int m = 1024;
    constexpr int n = 1024;
    constexpr int k = 1024;
    constexpr int warmup = 3;
    constexpr int iterations = 10;

    float *a = nullptr;
    float *b = nullptr;
    float *c = nullptr;

    cudaMalloc(&a, m * k * sizeof(float));
    cudaMalloc(&b, k * n * sizeof(float));
    cudaMalloc(&c, m * n * sizeof(float));

    for (int i = 0; i < warmup; ++i) {
        ccg::launch_naive_gemm_f32(a, b, c, m, n, k);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        ccg::launch_naive_gemm_f32(a, b, c, m, n, k);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    double average_ms = elapsed_ms / iterations;
    double tflops = 2.0 * m * n * k / average_ms / 1.0e9;

    std::printf("naive GEMM %dx%dx%d: %.3f ms, %.3f TFLOPS\n",
                m, n, k, average_ms, tflops);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    return 0;
}
