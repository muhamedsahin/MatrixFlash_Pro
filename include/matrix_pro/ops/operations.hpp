#pragma once

// Umbrella header for the whole operation surface of MatrixFlash-Pro.
//
// It aggregates the granular op headers so existing code that includes
// "matrix_pro/ops/operations.hpp" keeps working. New code is encouraged to
// include only the specific sub-header it needs (ops/reductions.hpp,
// ops/linalg.hpp, nn/conv.hpp, ...): the granular headers document their own
// semantics and keep compile times down.

#include <cstddef>
#include <vector>

#include "matrix_pro/core/matrix.hpp"
#include "matrix_pro/core/tensor.hpp"

// --- ops ---
#include "matrix_pro/ops/advanced.hpp"
#include "matrix_pro/ops/broadcast.hpp"
#include "matrix_pro/ops/elementwise.hpp"
#include "matrix_pro/ops/linalg.hpp"
#include "matrix_pro/ops/masking.hpp"
#include "matrix_pro/ops/precision.hpp"
#include "matrix_pro/ops/product.hpp"
#include "matrix_pro/ops/reductions.hpp"
#include "matrix_pro/ops/shape.hpp"
#include "matrix_pro/ops/transforms.hpp"

// --- nn ---
#include "matrix_pro/nn/activation.hpp"
#include "matrix_pro/nn/batch.hpp"
#include "matrix_pro/nn/conv.hpp"
#include "matrix_pro/nn/loss.hpp"
#include "matrix_pro/nn/ml.hpp"
