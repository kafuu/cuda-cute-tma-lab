add_library(ccg_basic_gemm STATIC src/gemm_naive.cu)
target_include_directories(ccg_basic_gemm PUBLIC include)
target_link_libraries(ccg_basic_gemm PUBLIC CUDA::cudart)

add_executable(test_gemm_naive tests/test_gemm_naive.cu)
target_link_libraries(test_gemm_naive PRIVATE ccg_basic_gemm)

add_executable(bench_gemm_naive bench/bench_gemm.cu)
target_link_libraries(bench_gemm_naive PRIVATE ccg_basic_gemm)

add_test(NAME basic_gemm.naive COMMAND test_gemm_naive)
