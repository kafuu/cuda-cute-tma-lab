# 3_tma_gemm

手写 CUDA TMA 路线。

当前目录提供 descriptor smoke test，验证：

- GPU compute capability 至少为 9.0
- CUDA driver API 可用
- `cuTensorMapEncodeTiled` 能成功编码一个 2D row-major tile

```bash
cmake --build --preset local-sm120-release --target test_tma_descriptor
ctest --preset local-sm120-release -R tma_gemm
```

后续 kernel 和 benchmark 放在本目录的 `src/`、`bench/` 中。TMA load、
mbarrier 和 pipeline 测试也放在 `tests/`。
