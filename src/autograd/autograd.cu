// Compilation entry point; implementation parts are private to this file.
// Keep this include order and translation unit to preserve inlining,
// kernel launch paths and shared caches. See the adjacent detail.txt.
#include "matrix_pro/autograd/autograd.hpp"
#include "matrix_pro/ops/operations.hpp"

#include <functional>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace matrix_pro {

#include "detail/node_types.inc"
#include "detail/tape_helpers.inc"
#include "variable/lifecycle.inc"
#include "variable/arithmetic.inc"
#include "variable/broadcast.inc"
#include "variable/shape.inc"
#include "variable/activations.inc"
#include "variable/losses.inc"

#include "tensor/lifecycle.inc"
#include "tensor/arithmetic.inc"
#include "tensor/cnn.inc"
#include "tensor/custom_ops.inc"
#include "detail/grad_mode.inc"
#include "variable/graph_control.inc"
}
