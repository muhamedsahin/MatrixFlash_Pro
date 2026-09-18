#pragma once

// Shape manipulation: reshape / concatenate / stack / scatter into a template
// matrix, plus one-hot encoding of integer label matrices.

#include <cstddef>
#include <vector>

#include "matrix_pro/core/matrix.hpp"

namespace matrix_pro {

// Row-major reshape; total element count must be preserved.
Matrix reshape(const Matrix& matrix, std::size_t rows, std::size_t cols);
// Concatenate along an axis: 0 = rows (vertical), 1 = cols (horizontal).
Matrix concat(const std::vector<Matrix>& parts, std::size_t axis);
// Stack matrices as rows of a single (n x rows*cols) matrix.
Matrix stack(const std::vector<Matrix>& parts);
// Writes `grad` into a zero matrix shaped like `like` at the given window.
Matrix slice_scatter(const Matrix& grad, const Matrix& like,
                     std::size_t row_start, std::size_t row_end,
                     std::size_t col_start, std::size_t col_end);
// One-hot rows for a label matrix: indices (n x 1) or (1 x n) -> (n x classes).
Matrix one_hot(const Matrix& indices, std::size_t classes);

} // namespace matrix_pro