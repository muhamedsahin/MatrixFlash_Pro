#pragma once

// Training-oriented GPU kernels: normalization layers and inverted dropout.
// Batch normalization normalizes each feature column, layer normalization
// normalizes each row.

#include "matrix_pro/core/matrix.hpp"

namespace matrix_pro {

Matrix batch_norm(const Matrix& matrix, float epsilon = 1e-5f);
Matrix layer_norm(const Matrix& matrix, float epsilon = 1e-5f);
Matrix dropout(const Matrix& matrix, float probability, unsigned long long seed = 0);

} // namespace matrix_pro