// Compilation entry point; implementation parts are private to this file.
// Keep this include order and translation unit to preserve inlining,
// kernel launch paths and shared caches. See the adjacent detail.txt.
#include "matrix_pro/ops/operations.hpp"
#include "matrix_pro/core/cuda_utils.hpp"

#include <cublas_v2.h>

#include <array>
#include <cmath>
#include <cfloat>
#include <limits>
#include <stdexcept>

namespace matrix_pro {
namespace {

#include "reductions/scratch.inc"
#include "reductions/scalar_kernels.cuh"
#include "reductions/axis_kernels.cuh"
#include "reductions/moment_kernels.cuh"
}

#include "reductions/scalars.inc"
#include "reductions/variance.inc"
#include "reductions/indices.inc"
#include "reductions/trace.inc"
#include "reductions/axes.inc"
#include "reductions/matrix_methods.inc"
#include "reductions/covariance.inc"
}