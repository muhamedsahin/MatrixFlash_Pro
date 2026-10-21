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

# A second add_subdirectory in the same build must not redefine the targets.
if(TARGET matrix_pro)
    return()
endif()

# Set the default before CUDA compiler detection initializes its own default.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# CUDA is optional. The CPU GEMM, thread pool, collectives and hybrid scheduler
# always build. The historical .cu sources and cuBLAS stay behind this switch.
if(NOT DEFINED MATRIX_PRO_ENABLE_CUDA)
	set(MATRIX_PRO_ENABLE_CUDA ON)
endif()
if(MATRIX_PRO_ENABLE_CUDA)
	project(MatrixFlashPro
	        VERSION 1.0.0
	        DESCRIPTION "Matrix and tensor library (CPU AVX, CUDA, collectives)"
	        LANGUAGES CXX CUDA)
else()
	project(MatrixFlashPro
	        VERSION 1.0.0
	        DESCRIPTION "Matrix and tensor library (CPU AVX, collectives)"
	        LANGUAGES CXX)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(MatrixFlashOptions)
include(MatrixFlashWarnings)

enable_testing()

# --- language / toolchain ---------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(MATRIX_PRO_ENABLE_CUDA)
	set(CMAKE_CUDA_STANDARD 17)
	set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

if(MSVC AND MATRIX_PRO_ENABLE_CUDA)
	# Keep the CUDA and host compilers' exception/optimization behaviour aligned.
	add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
endif()

if(MATRIX_PRO_ENABLE_CUDA)
	find_package(CUDAToolkit REQUIRED)
endif()

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
include(MatrixFlashSources)

set(MATRIX_PRO_HW_CXX ${MATRIX_PRO_HW_SOURCES})
set(MATRIX_PRO_HW_CU ${MATRIX_PRO_HW_SOURCES})
list(FILTER MATRIX_PRO_HW_CXX EXCLUDE REGEX "\\.cu$")
list(FILTER MATRIX_PRO_HW_CU INCLUDE REGEX "\\.cu$")

# CPU translation units stay out of /GL and /fp:fast so the host GEMM keeps
# IEEE rounding and the link does not need LTCG for these objects.
add_library(matrix_pro_cpu OBJECT ${MATRIX_PRO_HW_CXX})
target_include_directories(matrix_pro_cpu PUBLIC
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_features(matrix_pro_cpu PUBLIC cxx_std_17)
if(MATRIX_PRO_ENABLE_CUDA)
	target_compile_definitions(matrix_pro_cpu PUBLIC MATRIX_PRO_ENABLE_CUDA=1)
endif()
if(MSVC)
	target_compile_options(matrix_pro_cpu PRIVATE /W4 /permissive- /EHsc /utf-8)
	enable_language(ASM_MASM)
	target_sources(matrix_pro_cpu PRIVATE src/hw/gemm_avx2_kernel.asm)
else()
	target_compile_options(matrix_pro_cpu PRIVATE -Wall -Wextra)
endif()

if(MATRIX_PRO_ENABLE_CUDA)
	add_library(matrix_pro STATIC
	    ${MATRIX_PRO_LIBRARY_SOURCES}
	    ${MATRIX_PRO_IMPLEMENTATION_PARTS}
	    ${MATRIX_PRO_HW_CU}
	    $<TARGET_OBJECTS:matrix_pro_cpu>
	)
else()
	add_library(matrix_pro STATIC $<TARGET_OBJECTS:matrix_pro_cpu>)
endif()

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

if(MATRIX_PRO_ENABLE_CUDA)
	# cuBLAS (GEMM), cuSOLVER (decompositions), and cuSPARSE (sparse ops) are part of
	# the public API surface: consumers must inherit them.
	target_link_libraries(matrix_pro PUBLIC CUDA::cublas CUDA::cublasLt CUDA::cusolver CUDA::cusparse)
	target_compile_definitions(matrix_pro PUBLIC MATRIX_PRO_ENABLE_CUDA=1)
endif()

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
