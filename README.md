# CUDA-CuTe-TMA GEMM Lab

用同一个 GEMM 问题分别实现和验证 CUDA、CuTe DSL、TMA 三条路线，并在后续接入
小型 LLM 的线性层。

- 任务书：[TASKBOOK.md](TASKBOOK.md)
- 目标硬件：本地 RTX 5060 Ti（compute capability 12.0，8 GB）；H200 作为离线批次环境

## 目录

```text
0_introduction/    CUDA 入门、managed memory、访存和 shared memory 实验
1_basic_gemm/      naive 和 shared-memory CUDA GEMM
2_dsl_gemm/        CuTe Python DSL GEMM 与 Layout 实验
3_tma_gemm/        手写 TMA descriptor、load、mbarrier 和 pipeline
```

每个阶段目录都有自己的 `README.md`、`tests/`；需要 CUDA 的阶段有独立
`CMakeLists.txt`，DSL 阶段使用 Python 测试。

## 快速开始

```bash
cmake --preset local-sm120-debug
cmake --build --preset local-sm120-debug
ctest --preset local-sm120-debug

.venv/bin/pytest 2_dsl_gemm/tests
```

或者用统一脚本：

```bash
./scripts/run_tests.sh local-sm120-debug
```

## 当前状态

工程骨架已经建立。`1_basic_gemm` 当前包含可运行的 naive 基线；
`3_tma_gemm` 当前包含 descriptor smoke test；`2_dsl_gemm` 当前包含
Layout 参考实现和 DSL 环境检查。后续实现按阶段目录继续添加，不把不同方案
混在同一组 source/test 中。
