// Compilation entry point; implementation parts are private to this file.
// Keep this include order and translation unit to preserve inlining,
// kernel launch paths and shared caches. See the adjacent detail.txt.
#include "matrix_pro/core/cuda_utils.hpp"

#include <cublas_v2.h>
#include <cusolverDn.h>

#ifdef MATRIX_PRO_HAS_CUDNN
#include <cudnn.h>
#endif

#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <limits>

namespace matrix_pro {
namespace {

#include "runtime/execution_context.inc"
#include "runtime/solver_context.inc"
#include "runtime/pool_state.inc"

}

#include "runtime/allocation.inc"

#include "runtime/stream_access.inc"
#include "runtime/device_access.inc"
#include "runtime/solver_access.inc"

}
