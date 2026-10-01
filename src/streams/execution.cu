// Compilation entry point; implementation parts are private to this file.
// Keep this include order and translation unit to preserve inlining,
// kernel launch paths and shared caches. See the adjacent detail.txt.
#include "matrix_pro/streams/execution.hpp"
#include "matrix_pro/core/cuda_utils.hpp"
#include "matrix_pro/core/errors.hpp"
#include <cuda_runtime.h>
#include <sstream>

namespace matrix_pro {

#include "execution/graph.inc"
#include "execution/pipeline.inc"
#include "execution/profiler.inc"
} // namespace matrix_pro
