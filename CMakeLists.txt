cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(patchnorm LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

CUDA_ADD_LIBRARY(patchnorm SHARED
    patchnorm_cpu.cpp
    patchnorm_cuda.cu)

add_executable(patchnorm_test patchnorm.cpp )
target_link_libraries(patchnorm_test ${OpenCV_LIBS} patchnorm)
