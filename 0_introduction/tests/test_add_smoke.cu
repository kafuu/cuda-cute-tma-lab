#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>

__global__
void add_grid_stride(const float *x, const float *y, float *out, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride) {
        out[i] = x[i] + y[i];
    }
}

int main()
{
    constexpr int n = 1000003;
    constexpr size_t bytes = n * sizeof(float);

    float *x = nullptr;
    float *y = nullptr;
    float *out = nullptr;

    cudaMallocManaged(&x, bytes);
    cudaMallocManaged(&y, bytes);
    cudaMallocManaged(&out, bytes);

    for (int i = 0; i < n; ++i) {
        x[i] = static_cast<float>(i % 17);
        y[i] = static_cast<float>(i % 13);
    }

    add_grid_stride<<<256, 256>>>(x, y, out, n);
    cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        std::fprintf(stderr, "kernel failed: %s\n", cudaGetErrorString(error));
        return 1;
    }

    float max_error = 0.0f;
    for (int i = 0; i < n; ++i) {
        float expected = static_cast<float>(i % 17 + i % 13);
        max_error = std::fmax(max_error, std::fabs(out[i] - expected));
    }

    cudaFree(x);
    cudaFree(y);
    cudaFree(out);

    std::printf("introduction add max error: %.8f\n", max_error);
    return max_error == 0.0f ? 0 : 1;
}
