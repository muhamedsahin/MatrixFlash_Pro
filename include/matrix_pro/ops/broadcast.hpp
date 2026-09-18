#pragma once

// Broadcasting: the classic row/column-vector helpers (`v` concatenated onto
// every row / column) and the general 2D broadcast family that NumPy-style
// shapes such as (M,1) op (1,N) -> (M,N) require.

#include "matrix_pro/core/matrix.hpp"

namespace matrix_pro {

// --- vector broadcast ---
Matrix add_row_vector(const Matrix& matrix, const Matrix& vector);       // v length == cols
Matrix add_col_vector(const Matrix& matrix, const Matrix& vector);       // v length == rows
Matrix multiply_row_vector(const Matrix& matrix, const Matrix& vector);
Matrix multiply_col_vector(const Matrix& matrix, const Matrix& vector);

// --- general 2D broadcasting ---
// Each dimension must match or be 1 on one side ((M,1) + (1,N) -> (M,N) etc).
Matrix broadcast_add(const Matrix& left, const Matrix& right);
Matrix broadcast_subtract(const Matrix& left, const Matrix& right);
Matrix broadcast_multiply(const Matrix& left, const Matrix& right);
Matrix broadcast_divide(const Matrix& left, const Matrix& right);
// Gradient of a broadcast operand: sums grad over the dimensions that were
// broadcast, producing a matrix shaped like `target`.
Matrix broadcast_backward(const Matrix& grad, const Matrix& target);

} // namespace matrix_pro