#pragma once

// Square-matrix scalar/derived quantities that are not full decompositions.
// (The decompositions themselves live in matrix_pro/ops/linalg.hpp.)

#include "matrix_pro/core/matrix.hpp"

namespace matrix_pro {

// LU determinant of a square matrix.
float determinant(const Matrix& matrix);
// Matrix inverse for a square, non-singular matrix (getrf + getri).
Matrix inverse(const Matrix& matrix);
// ||A|| * ||A^-1|| (1-norm estimate), a cheap conditioning indicator.
float condition_number(const Matrix& matrix);
// Covariance / correlation matrices treating each row as an observation.
Matrix covariance(const Matrix& matrix);
Matrix correlation(const Matrix& matrix);

} // namespace matrix_pro