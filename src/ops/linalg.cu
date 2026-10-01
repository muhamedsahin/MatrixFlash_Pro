// Compilation entry point; implementation parts are private to this file.
// Keep this include order and translation unit to preserve inlining,
// kernel launch paths and shared caches. See the adjacent detail.txt.
#include "matrix_pro/ops/operations.hpp"
#include "matrix_pro/core/cuda_utils.hpp"

#include <cusolverDn.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace matrix_pro {
#include "linalg/layout_kernels.cuh"
#include "linalg/solve.inc"
#include "linalg/qr.inc"
#include "linalg/svd.inc"
#include "linalg/cholesky.inc"
#include "linalg/eigen.inc"
#include "linalg/derived_solvers.inc"
#include "linalg/matrix_methods.inc"
} // namespace matrix_pro