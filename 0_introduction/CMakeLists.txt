add_executable(intro_add_smoke tests/test_add_smoke.cu)
target_link_libraries(intro_add_smoke PRIVATE CUDA::cudart)

add_test(NAME introduction.add_smoke COMMAND intro_add_smoke)
