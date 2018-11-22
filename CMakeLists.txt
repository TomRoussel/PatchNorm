cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(patchnorm LANGUAGES CXX)

set (CMAKE_CXX_STANDARD 11)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED core highgui imgcodecs)

include_directories(${OpenCV_INCLUDE_DIRS})

CUDA_ADD_LIBRARY(patchnorm SHARED
    patchnorm_cpu.cpp
    patchnorm_cuda.cu)

target_link_libraries(patchnorm ${OpenCV_LIBS})

add_executable(patchnorm_convert convert.cpp)
target_link_libraries(patchnorm_convert ${OpenCV_LIBS} patchnorm)
add_executable(patchnorm_test patchnorm.cpp)
target_link_libraries(patchnorm_test ${OpenCV_LIBS} patchnorm)
