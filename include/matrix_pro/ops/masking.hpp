#pragma once

// Comparison, logical and masking operations. Comparison kernels return 1.0f /
// 0.0f masks so their output composes directly with the arithmetic ops.

#include "matrix_pro/core/matrix.hpp"

namespace matrix_pro {

// --- comparison ---
Matrix greater(const Matrix& matrix, float scalar);
Matrix greater(const Matrix& left, const Matrix& right);
Matrix less(const Matrix& matrix, float scalar);
Matrix less(const Matrix& left, const Matrix& right);
Matrix equal(const Matrix& matrix, float scalar);
Matrix equal(const Matrix& left, const Matrix& right);
Matrix not_equal(const Matrix& matrix, float scalar);
Matrix not_equal(const Matrix& left, const Matrix& right);

// --- logical ---
Matrix logical_and(const Matrix& left, const Matrix& right);
Matrix logical_or(const Matrix& left, const Matrix& right);
Matrix logical_not(const Matrix& matrix);
Matrix isnan(const Matrix& matrix);
Matrix isinf(const Matrix& matrix);
Matrix is_finite(const Matrix& matrix);
bool any(const Matrix& matrix);
bool all(const Matrix& matrix);

// --- selection / masking ---
Matrix where(const Matrix& condition, const Matrix& true_value, const Matrix& false_value);
Matrix where(const Matrix& condition, float true_value, float false_value);
// Copy of `matrix` with every `mask != 0` entry replaced by `value`.
Matrix apply_mask(const Matrix& matrix, const Matrix& mask, float value);
// Compacted (1 x count) copy holding only the entries selected by `mask`.
Matrix filter_by_mask(const Matrix& matrix, const Matrix& mask);

} // namespace matrix_pro