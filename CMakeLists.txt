cmake_minimum_required(VERSION 3.24)
project(MatrixFlashPro LANGUAGES CXX CUDA)

enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native)

find_package(CUDAToolkit REQUIRED)

add_library(matrix_pro STATIC
	src/cuda_utils.cu
	src/matrix.cu
	src/matrix_factories.cu
	src/operations_elementwise.cu
	src/operations_matmul.cu
	src/operations_transforms.cu
	src/operations_statistics.cu
	src/operations_advanced.cu
)

target_include_directories(matrix_pro PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(matrix_pro PUBLIC CUDA::cublas CUDA::cusolver)
set_target_properties(matrix_pro PROPERTIES
	CUDA_SEPARABLE_COMPILATION ON
	CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

add_executable(matrix_pro_basic examples/basic_gpu_usage.cpp)
target_link_libraries(matrix_pro_basic PRIVATE matrix_pro)

add_subdirectory(tests)

add_executable(matrix_pro_benchmark benchmarks/perf_matrix.cpp)
target_link_libraries(matrix_pro_benchmark PRIVATE matrix_pro)
