// Compilation entry point; implementation parts are private to this file.
// Keep this include order and translation unit to preserve inlining,
// kernel launch paths and shared caches. See the adjacent detail.txt.
#include "matrix_pro/ops/operations.hpp"
#include "matrix_pro/core/cuda_utils.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cfloat>
#include <stdexcept>

namespace matrix_pro {
#include "backward/convolution_kernels.cuh"

namespace {

#include "backward/pooling_kernels.cuh"
#include "backward/tensor_kernels.cuh"
}

#include "backward/convolution.inc"
#include "backward/pooling.inc"
#include "backward/tensor_arithmetic.inc"
#include "backward/batched_matmul.inc"
}  // namespace matrix_pro

