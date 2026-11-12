# 2_dsl_gemm

CuTe Python DSL 路线。该目录不进入 CMake 的 CUDA 构建，测试使用 Python。

当前内容：

- `src/dsl_gemm/layout_reference.py`：用纯 Python 表达 `(shape, stride)` 和行主序地址。
- `tests/test_layout_reference.py`：固定手工计算结果。
- `tests/test_dsl_environment.py`：检查 `nvidia-cutlass-dsl` 是否可用；未安装时 skip。

```bash
.venv/bin/pytest 2_dsl_gemm/tests
```

安装 DSL 后，新的 kernel 和测试继续放在本目录的 `src/`、`tests/` 中，不要混入
`1_basic_gemm` 或 `3_tma_gemm` 的 CMake target。
