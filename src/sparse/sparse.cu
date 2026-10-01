// Compilation entry point; implementation parts are private to this file.
// Keep this include order and translation unit to preserve inlining,
// kernel launch paths and shared caches. See the adjacent detail.txt.
#include "matrix_pro/sparse/sparse.hpp"
#include "matrix_pro/core/matrix.hpp"
#include "matrix_pro/core/cuda_utils.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include <limits>

namespace matrix_pro {
#include "csr/kernels.cuh"

#include "csr/construction.inc"
#include "csr/storage.inc"
#include "csr/products.inc"
} // namespace matrix_pro
