# ============================================================================
#  MatrixFlash-Pro — GPU matrix / tensor library
#  ---------------------------------------------------------------------------
#  Layout
#    include/matrix_pro/   public headers, one folder per module
#    src/                  implementation, mirrors include/matrix_pro
#    tests/                unit + integration behavior tests (CTest)
#    benchmarks/           performance suite
#    examples/             runnable usage examples
#    cmake/                build modules (options, warnings, install)
#  ---------------------------------------------------------------------------
#  Typical use
#    cmake --preset release
#    cmake --build --preset release
#    ctest --preset release-tests
# ============================================================================

cmake_minimum_required(VERSION 3.24)

project(MatrixFlashPro
        VERSION 1.0.0
        DESCRIPTION "GPU-accelerated matrix / tensor library (CUDA, cuBLAS, cuSOLVER)"
        LANGUAGES CXX CUDA)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(MatrixFlashOptions)
include(MatrixFlashWarnings)

enable_testing()

# --- language / toolchain ---------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# Target the GPU actually installed in the build machine.
set(CMAKE_CUDA_ARCHITECTURES native)

if(MSVC)
	# Keep the CUDA and host compilers' exception/optimization behaviour aligned.
	add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
endif()

find_package(CUDAToolkit REQUIRED)

# --- optional cuDNN backend for convolution / pooling -----------------------
set(MATRIX_PRO_CUDNN_FOUND FALSE)
if(MATRIX_PRO_ENABLE_CUDNN)
	find_path(MATRIX_PRO_CUDNN_INCLUDE_DIR cudnn.h
	          HINTS "$ENV{CUDA_PATH}/include")
	find_library(MATRIX_PRO_CUDNN_LIBRARY cudnn
	             HINTS "$ENV{CUDA_PATH}/lib/x64" "$ENV{CUDA_PATH}/lib64")
	if(MATRIX_PRO_CUDNN_INCLUDE_DIR AND MATRIX_PRO_CUDNN_LIBRARY)
		set(MATRIX_PRO_CUDNN_FOUND TRUE)
	endif()
endif()

# ============================================================================
#  matrix_pro library
# ============================================================================
add_library(matrix_pro STATIC
	# --- core ---
	src/core/cuda_utils.cu
	src/core/matrix.cu
	src/core/matrix_factories.cu
	src/core/tensor.cu
	# --- ops ---
	src/ops/elementwise.cu
	src/ops/matmul.cu
	src/ops/gemm/gemm_dispatch.cu
	src/ops/gemm/gemm_cublas_ex.cu
	src/ops/gemm/gemm_cublas_lt.cu
	src/ops/gemm/gemm_micro.cu
	src/ops/gemm/gemm_gemv.cu
	src/ops/gemm/gemm_tiled.cu
	src/ops/gemm/gemm_fused.cu
	src/ops/transforms.cu
	src/ops/statistics.cu
	src/ops/shape.cu
	src/ops/dtype.cu
	src/ops/advanced.cu
	src/ops/linalg.cu
	src/ops/masking.cu
	src/ops/extended.cu
	src/ops/precision.cu
	src/ops/view.cu
	# --- nn ---
	src/nn/activation.cu
	src/nn/loss.cu
	src/nn/conv_backward.cu
	src/nn/batch.cu
	src/nn/ml.cu
	src/nn/conv.cu
	src/nn/fused.cu
	src/nn/inplace.cu
	# --- autograd / sparse / indexing / rng / streams ---
	src/autograd/autograd.cu
	src/sparse/sparse.cu
	src/indexing/indexing.cu
	src/rng/rng.cu
	src/streams/streams.cu
)

add_library(matrix_pro::matrix_pro ALIAS matrix_pro)

set_target_properties(matrix_pro PROPERTIES
	OUTPUT_NAME matrix_pro
	CXX_STANDARD 17
	CXX_STANDARD_REQUIRED ON
)

target_compile_features(matrix_pro PUBLIC cxx_std_17)

target_include_directories(matrix_pro
	PUBLIC
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
		$<INSTALL_INTERFACE:include>
)

# cuBLAS (GEMM) and cuSOLVER (decompositions) are part of the public API surface:
# consumers must inherit them.
target_link_libraries(matrix_pro PUBLIC CUDA::cublas CUDA::cublasLt CUDA::cusolver)

if(MATRIX_PRO_CUDNN_FOUND)
	# PUBLIC, not PRIVATE: core/cuda_utils.hpp conditionally declares
	# cudnn_handle() under MATRIX_PRO_HAS_CUDNN, so consumers must see exactly the
	# same definition *and* the cuDNN include path, otherwise the API surface
	# differs between the library and its users.
	target_compile_definitions(matrix_pro PUBLIC MATRIX_PRO_HAS_CUDNN=1)
	target_include_directories(matrix_pro PUBLIC ${MATRIX_PRO_CUDNN_INCLUDE_DIR})
	target_link_libraries(matrix_pro PUBLIC ${MATRIX_PRO_CUDNN_LIBRARY})
	message(STATUS "MatrixFlash-Pro: cuDNN backend enabled (${MATRIX_PRO_CUDNN_LIBRARY})")
elseif(MATRIX_PRO_ENABLE_CUDNN)
	message(STATUS "MatrixFlash-Pro: cuDNN not found; using CUDA convolution/pooling fallback")
else()
	message(STATUS "MatrixFlash-Pro: cuDNN backend disabled by option")
endif()

matrix_pro_set_warnings(matrix_pro)

# --- aggressive release optimizations ---------------------------------------
# Host side (MSVC): whole-program optimization + fast FP.
# Device side: ptxas max effort; optional --use_fast_math for the SFU path.
if(MSVC)
	target_compile_options(matrix_pro PRIVATE
		$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/GL /fp:fast /Ob3>)
endif()
if(CMAKE_CUDA_COMPILER_ID MATCHES "NVIDIA")
	target_compile_options(matrix_pro PRIVATE
		$<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-O3,-allow-expensive-optimizations=true>)
	if(MATRIX_PRO_FAST_MATH)
		target_compile_options(matrix_pro PRIVATE
			$<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
	endif()
endif()

# ============================================================================
#  Tests / benchmarks / examples
# ============================================================================
if(MATRIX_PRO_BUILD_TESTS)
	add_subdirectory(tests)
endif()

if(MATRIX_PRO_BUILD_BENCHMARKS)
	add_subdirectory(benchmarks)
endif()

if(MATRIX_PRO_BUILD_EXAMPLES)
	add_subdirectory(examples)
endif()

include(MatrixFlashInstall)

message(STATUS "MatrixFlash-Pro ${PROJECT_VERSION}: tests=${MATRIX_PRO_BUILD_TESTS} "
               "benchmarks=${MATRIX_PRO_BUILD_BENCHMARKS} examples=${MATRIX_PRO_BUILD_EXAMPLES} "
               "cudnn=${MATRIX_PRO_CUDNN_FOUND}")
