#pragma once

// Reductions: axis reductions producing matrices plus device-side scalar
// reductions (single kernel + one host read, no intermediate copies).

#include <cstddef>

#include "matrix_pro/core/matrix.hpp"

namespace matrix_pro {

// --- axis reductions ---
Matrix row_sum(const Matrix& matrix);  // (rows x 1)
Matrix col_sum(const Matrix& matrix);  // (1 x cols)

// --- scalar reductions (GPU-accelerated) ---
float sum(const Matrix& matrix);
float mean(const Matrix& matrix);
float min(const Matrix& matrix);
float max(const Matrix& matrix);
std::size_t argmin(const Matrix& matrix);
std::size_t argmax(const Matrix& matrix);
float variance(const Matrix& matrix);
float stddev(const Matrix& matrix);
float l1_norm(const Matrix& matrix);
float l2_norm(const Matrix& matrix);
float frobenius_norm(const Matrix& matrix);
float abs_max(const Matrix& matrix);
float trace(const Matrix& matrix);

} // namespace matrix_pro