// Compilation entry point; implementation parts are private to this file.
// Keep this include order and translation unit to preserve inlining,
// kernel launch paths and shared caches. See the adjacent detail.txt.
#include "matrix_pro/ops/operations.hpp"
#include "matrix_pro/core/cuda_utils.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace matrix_pro {
#include "masking/kernels.cuh"
#include "masking/comparisons.inc"
#include "masking/predicates.inc"
#include "masking/selection.inc"

#include "masking/matrix_methods.inc"
} // namespace matrix_pro
