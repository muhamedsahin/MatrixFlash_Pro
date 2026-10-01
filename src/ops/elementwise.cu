// Compilation entry point; implementation parts are private to this file.
// Keep this include order and translation unit to preserve inlining,
// kernel launch paths and shared caches. See the adjacent detail.txt.
#include "matrix_pro/ops/operations.hpp"
#include "matrix_pro/core/cuda_utils.hpp"
#include "matrix_pro/detail/kernel_helpers.cuh"

#include <stdexcept>

namespace matrix_pro {
namespace {

#include "elementwise/functors.cuh"
#include "elementwise/broadcast_kernels.cuh"

#include "elementwise/unary_launch.inc"
}

#include "elementwise/arithmetic.inc"
#include "elementwise/broadcast.inc"
#include "elementwise/matrix_methods.inc"
}

