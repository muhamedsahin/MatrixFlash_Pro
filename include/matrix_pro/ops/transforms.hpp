#pragma once

// Whole-tensor layout transforms and the "classic" activation kernels that the
// Matrix convenience methods delegate to (transpose/relu/softmax/...).
// Zero-copy, strided alternatives live in matrix_pro/view/view.hpp.

#include <cstddef>

#include "matrix_pro/core/matrix.hpp"

namespace matrix_pro {

Matrix transpose(const Matrix& matrix);
Matrix relu(const Matrix& matrix);
Matrix softmax(const Matrix& matrix);   // row-wise, numerically stable
Matrix flatten(const Matrix& matrix);
Matrix slice(const Matrix& matrix, std::size_t row_start, std::size_t row_end,
             std::size_t col_start, std::size_t col_end);

} // namespace matrix_pro
