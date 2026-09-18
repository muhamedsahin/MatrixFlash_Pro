# Central build options for MatrixFlash-Pro.
# Include this module before any add_subdirectory()/add_library() call.

include_guard(GLOBAL)

option(MATRIX_PRO_BUILD_TESTS
       "Build the MatrixFlash-Pro behavior test suite" ON)
option(MATRIX_PRO_BUILD_EXAMPLES
       "Build the MatrixFlash-Pro example programs" ON)
option(MATRIX_PRO_BUILD_BENCHMARKS
       "Build the MatrixFlash-Pro benchmark suite" ON)
option(MATRIX_PRO_ENABLE_CUDNN
       "Use cuDNN for convolution/pooling when the toolkit is available" ON)
option(MATRIX_PRO_WARNINGS
       "Enable the project warning set for C++ translation units" ON)
option(MATRIX_PRO_CUDA_WARNINGS
       "Also forward the warning set to NVCC host compilation (noisy: off by default)" OFF)
option(MATRIX_PRO_INSTALL
       "Generate install/export rules (find_package(matrix_pro) support)" ON)
option(MATRIX_PRO_FAST_MATH
       "Build CUDA kernels with -use_fast_math (faster SFU math, slightly lower ULP accuracy)" OFF)