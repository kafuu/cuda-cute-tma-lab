# 0_introduction

CUDA 入门示例和基础行为验证。

现有子目录保留按主题整理的实验代码：

- `0.1 An Even Easier Introduction to CUDA`
- `0.2 Global Memory Access in CUDA`
- `0.3 Shared memory in CUDA`

`tests/` 中放可被根 CMake 和 CTest 自动执行的 smoke test。当前测试覆盖
managed memory、线程索引和 grid-stride loop。

```bash
cmake --build --preset local-sm120-release --target intro_add_smoke
ctest --preset local-sm120-release -R introduction
```
