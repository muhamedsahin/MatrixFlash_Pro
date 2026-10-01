// Compilation entry point; implementation parts are private to this file.
// Keep this include order and translation unit to preserve inlining,
// kernel launch paths and shared caches. See the adjacent detail.txt.
#include "matrix_pro/ops/linalg_extended.hpp"
#include "matrix_pro/core/matrix.hpp"
#include "matrix_pro/core/tensor.hpp"
#include "matrix_pro/core/cuda_utils.hpp"
#include "matrix_pro/core/errors.hpp"
#include "matrix_pro/ops/linalg.hpp"
#include "matrix_pro/ops/operations.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <vector>
#include <cmath>
#include <limits>
#include "matrix_pro/ops/gemm.hpp"
#include "matrix_pro/ops/factorization.hpp"

namespace matrix_pro {

#include "linalg/extended_kernels.cuh"

#include "linalg/triangular.inc"
#include "linalg/matrix_functions.inc"
#include "linalg/batched_solvers.inc"
} // namespace matrix_pro
