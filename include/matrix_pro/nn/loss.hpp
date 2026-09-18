#pragma once

// Fused, numerically stable loss primitives (single kernel per call).
// The `_value` functions return the scalar loss, the `_grad` functions return
// dL/dz shaped like the inputs.

#include "matrix_pro/core/matrix.hpp"

namespace matrix_pro {

// Binary cross-entropy with logits: mean over elements of
// max(z,0) - z*t + log(1 + exp(-|z|)).
float bce_with_logits_value(const Matrix& logits, const Matrix& target);
Matrix bce_with_logits_grad(const Matrix& logits, const Matrix& target);

// Categorical cross-entropy over rows: logits (rows x classes), target one-hot
// (rows x classes). Returns per-row loss (rows x 1) / gradient dL/dlogits.
Matrix softmax_cross_entropy_value(const Matrix& logits, const Matrix& target);
Matrix softmax_cross_entropy_grad(const Matrix& logits, const Matrix& target);

} // namespace matrix_pro