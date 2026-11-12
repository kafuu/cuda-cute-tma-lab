#pragma once

#include <cuda_runtime.h>

namespace ccg {

cudaError_t launch_naive_gemm_f32(
    const float *a,
    const float *b,
    float *c,
    int m,
    int n,
    int k);

}  // namespace ccg
