// Compilation entry point; implementation parts are private to this file.
// Keep this include order and translation unit to preserve inlining,
// kernel launch paths and shared caches. See the adjacent detail.txt.
#include "matrix_pro/core/tensor.hpp"
#include "matrix_pro/core/cuda_utils.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace matrix_pro {
#include "tensor/storage_kernels.cuh"
#include "tensor/storage.inc"
#include "tensor/transfers.inc"
#include "tensor/contiguous.inc"
#include "tensor/views.inc"
#include "tensor/slicing.inc"

}