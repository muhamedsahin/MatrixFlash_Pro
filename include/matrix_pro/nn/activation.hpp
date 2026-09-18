#pragma once

// Neural-network activation functions with their backward kernels, including
// the numerically stable softplus/mish family that needs a fused kernel.

#include "matrix_pro/core/matrix.hpp"

namespace matrix_pro {

// --- forward ---
Matrix softplus(const Matrix& matrix);
Matrix mish(const Matrix& matrix);
Matrix hardtanh(const Matrix& matrix, float low = -1.0f, float high = 1.0f);
Matrix hardsigmoid(const Matrix& matrix);
Matrix hardswish(const Matrix& matrix);
Matrix selu(const Matrix& matrix);
Matrix prelu(const Matrix& matrix, const Matrix& alpha);   // alpha: 1x1 or rows x cols

// --- backward (dL/dx given dL/df and the forward input) ---
Matrix softplus_backward(const Matrix& input, const Matrix& grad);
Matrix mish_backward(const Matrix& input, const Matrix& grad);

} // namespace matrix_pro