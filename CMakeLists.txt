cmake_minimum_required(VERSION 3.24)

project(cuda_cute_tma_lab LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "CUDA architectures")
endif()

find_package(CUDAToolkit REQUIRED)

include(CTest)

add_subdirectory(0_introduction)
add_subdirectory(1_basic_gemm)
add_subdirectory(3_tma_gemm)
