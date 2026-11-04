# CUDA-CuTe-TMA GEMM Lab

用同一个 LLM GEMM 问题，依次实现 naive CUDA、shared-memory tiled CUDA、TMA 和 CuTe 四条数据路径，把通过的实现接入一个小型 LLM，完成算子级、模块级和端到端验证。

- 任务书：[TASKBOOK.md](TASKBOOK.md)（当前版本 0.4）
- 目标硬件：本地 RTX 5060 Ti（compute capability 12.0，8 GB）；H200（sm_90 / sm_90a）作为离线批次环境

## 状态

P0 未开始。本机尚未安装 git、CMake、Ninja、MSVC、CUDA Toolkit、PyTorch。
