cmake_minimum_required(VERSION 3.24)
project(MatrixFlashPro LANGUAGES CXX CUDA)

enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native)

find_package(CUDAToolkit REQUIRED)
find_path(MATRIX_PRO_CUDNN_INCLUDE_DIR cudnn.h
	HINTS "$ENV{CUDA_PATH}/include")
find_library(MATRIX_PRO_CUDNN_LIBRARY cudnn
	HINTS "$ENV{CUDA_PATH}/lib/x64" "$ENV{CUDA_PATH}/lib64")

add_library(matrix_pro STATIC
	src/cuda_utils.cu
	src/matrix.cu
	src/matrix_factories.cu
	src/operations_elementwise.cu
	src/operations_matmul.cu
	src/operations_transforms.cu
	src/operations_statistics.cu
	src/operations_advanced.cu
	src/operations_linalg.cu
	src/operations_masking.cu
	src/operations_batch.cu
	src/operations_extended.cu
	src/operations_precision.cu
	src/operations_ml.cu
	src/operations_conv.cu
	src/autograd.cu
	src/tensor.cu
)

target_link_libraries(matrix_pro PUBLIC CUDA::cublas CUDA::cusolver)

if (MATRIX_PRO_CUDNN_INCLUDE_DIR AND MATRIX_PRO_CUDNN_LIBRARY)
	target_compile_definitions(matrix_pro PRIVATE MATRIX_PRO_HAS_CUDNN=1)
	target_include_directories(matrix_pro PRIVATE ${MATRIX_PRO_CUDNN_INCLUDE_DIR})
	target_link_libraries(matrix_pro PUBLIC ${MATRIX_PRO_CUDNN_LIBRARY})
message(STATUS "MatrixFlash-Pro: cuDNN backend enabled")
else()
	message(STATUS "MatrixFlash-Pro: cuDNN not found; using CUDA convolution/pooling fallback")
endif()

target_include_directories(matrix_pro
	PUBLIC
		${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(matrix_pro_basic examples/basic_gpu_usage.cpp)
target_link_libraries(matrix_pro_basic PRIVATE matrix_pro)

add_subdirectory(tests)

add_executable(matrix_pro_benchmark benchmarks/perf_matrix.cpp)
target_link_libraries(matrix_pro_benchmark PRIVATE matrix_pro)
