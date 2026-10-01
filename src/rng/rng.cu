// Compilation entry point; implementation parts are private to this file.
// Keep this include order and translation unit to preserve inlining,
// kernel launch paths and shared caches. See the adjacent detail.txt.
#include "matrix_pro/rng/rng.hpp"
#include "matrix_pro/core/matrix.hpp"
#include "matrix_pro/core/cuda_utils.hpp"

#include <cuda_runtime.h>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <utility>

namespace matrix_pro {
namespace {

#include "detail/counter_state.inc"
#include "detail/base_kernels.cuh"
#include "detail/counter_access.inc"
} // namespace

#include "distributions/basic.inc"
namespace {
#include "detail/distribution_kernels.cuh"
} // namespace

#include "distributions/extended.inc"
#include "distributions/state.inc"
} // namespace matrix_pro
