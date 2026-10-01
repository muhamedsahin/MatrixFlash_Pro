// Compilation entry point; implementation parts are private to this file.
// Keep this include order and translation unit to preserve inlining,
// kernel launch paths and shared caches. See the adjacent detail.txt.
#include "matrix_pro/sparse/sparse_formats.hpp"
#include "matrix_pro/sparse/sparse.hpp"
#include "matrix_pro/core/matrix.hpp"
#include "matrix_pro/core/cuda_utils.hpp"
#include "matrix_pro/core/errors.hpp"
#include "matrix_pro/detail/kernel_helpers.cuh"

#include <cuda_runtime.h>
#include <cusparse.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <map>
#include <limits>

namespace matrix_pro {

#include "formats/kernels.cuh"

#include "formats/coo.inc"
#include "formats/csc.inc"
#include "formats/conversions.inc"
#include "formats/spgemm.inc"
#include "formats/addition.inc"
} // namespace matrix_pro
