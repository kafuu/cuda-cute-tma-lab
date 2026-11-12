#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>

namespace {

const char *driver_error(CUresult result)
{
    const char *message = nullptr;
    cuGetErrorString(result, &message);
    return message != nullptr ? message : "unknown CUDA driver error";
}

}  // namespace

int main()
{
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::printf("no CUDA device; skipping TMA descriptor test\n");
        return 77;
    }

    int major = 0;
    int minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0);
    if (major < 9) {
        std::printf("compute capability %d.%d has no TMA; skipping\n", major, minor);
        return 77;
    }

    cudaFree(0);

    constexpr int rows = 128;
    constexpr int cols = 256;
    void *global = nullptr;
    cudaError_t cuda_status =
        cudaMalloc(&global, rows * cols * sizeof(float));
    if (cuda_status != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc failed: %s\n",
                     cudaGetErrorString(cuda_status));
        return 1;
    }

    cuuint64_t global_dim[2] = {cols, rows};
    cuuint64_t global_strides[1] = {cols * sizeof(float)};
    cuuint32_t box_dim[2] = {32, 32};
    cuuint32_t element_strides[2] = {1, 1};

    alignas(64) CUtensorMap tensor_map{};
    CUresult result = cuTensorMapEncodeTiled(
        &tensor_map,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        2,
        global,
        global_dim,
        global_strides,
        box_dim,
        element_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    cudaFree(global);

    if (result != CUDA_SUCCESS) {
        std::fprintf(stderr, "cuTensorMapEncodeTiled failed: %s\n",
                     driver_error(result));
        return 1;
    }

    std::printf(
        "TMA descriptor encoded: dtype=float32, global=%dx%d, box=32x32\n",
        cols,
        rows);
    return 0;
}
