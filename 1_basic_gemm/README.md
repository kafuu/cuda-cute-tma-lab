# 1_basic_gemm

不使用 DSL 或 TMA 的基础 CUDA GEMM 路线。

规划中的实现：

- `gemm_naive.cu`：一个线程计算一个 C 元素
- `gemm_shared.cu`：shared-memory tiled GEMM
- `gemm_register.cu`：寄存器分块和 thread coarsening

当前目录提供可运行的 naive 基线，用来验证以下约定：

```text
A: M x K, row-major
B: K x N, row-major
C: M x N, row-major

C[row * N + col] =
    sum_k A[row * K + k] * B[k * N + col]
```

```bash
cmake --build --preset local-sm120-release --target test_gemm_naive
ctest --preset local-sm120-release -R basic_gemm
./build/local-sm120-release/1_basic_gemm/bench_gemm_naive
```
