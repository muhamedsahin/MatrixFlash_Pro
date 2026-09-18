#pragma once

// In-place elementwise / activation variants (add_, relu_, ...).
// They reuse the same vectorized kernels as the out-of-place ops but write
// back into the input buffer, saving one allocation + one full memory pass
// per call -- critical for large-model training loops.

#include <cstddef>

namespace matrix_pro {

class Matrix;

Matrix& add_(Matrix& self, const Matrix& other);
Matrix& subtract_(Matrix& self, const Matrix& other);
Matrix& elementwise_multiply_(Matrix& self, const Matrix& other);
Matrix& divide_(Matrix& self, const Matrix& other);
Matrix& add_scalar_(Matrix& self, float value);
Matrix& multiply_scalar_(Matrix& self, float value);
Matrix& negate_(Matrix& self);
Matrix& clamp_(Matrix& self, float low, float high);
Matrix& relu_(Matrix& self);
Matrix& sigmoid_(Matrix& self);
Matrix& tanh_(Matrix& self);
Matrix& leaky_relu_(Matrix& self, float negative_slope = 0.01f);
Matrix& elu_(Matrix& self, float alpha = 1.0f);
Matrix& gelu_(Matrix& self);
Matrix& swish_(Matrix& self, float beta = 1.0f);

// Broadcast-aware in-place add/mul: rhs may be (1x1), (1xC), (Rx1) or (RxC).
Matrix& broadcast_add_(Matrix& self, const Matrix& rhs);
Matrix& broadcast_multiply_(Matrix& self, const Matrix& rhs);

} // namespace matrix_pro
