add_executable(test_tma_descriptor tests/test_tma_descriptor.cu)
target_link_libraries(test_tma_descriptor PRIVATE CUDA::cudart CUDA::cuda_driver)

add_test(NAME tma_gemm.descriptor COMMAND test_tma_descriptor)
set_tests_properties(tma_gemm.descriptor PROPERTIES SKIP_RETURN_CODE 77)
