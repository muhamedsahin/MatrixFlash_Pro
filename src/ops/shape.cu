// Compilation entry point; implementation parts are private to this file.
// Keep this include order and translation unit to preserve inlining,
// kernel launch paths and shared caches. See the adjacent detail.txt.
#include "matrix_pro/ops/operations.hpp"
#include "matrix_pro/core/cuda_utils.hpp"
#include "matrix_pro/detail/kernel_helpers.cuh"

#include <cuda_runtime.h>
#include <algorithm>
#include <stdexcept>

namespace matrix_pro {
#include "shape/kernels.cuh"

#include "shape/broadcast.inc"
#include "shape/reshape_slice.inc"
#include "shape/concatenate.inc"
}
