#pragma once

// Elementwise arithmetic, elementwise math and scalar helpers.
// Every function here is an out-of-place GPU kernel: the inputs are never
// modified. For the allocation-free in-place variants (`relu_`, `add_`, ...)
// see matrix_pro/nn/inplace.hpp.

#include <cstddef>

#include "matrix_pro/core/matrix.hpp"

namespace matrix_pro {

// --- basic elementwise ---
Matrix add(const Matrix& left, const Matrix& right);
Matrix subtract(const Matrix& left, const Matrix& right);
Matrix elementwise_multiply(const Matrix& left, const Matrix& right);   // Hadamard
Matrix multiply(const Matrix& matrix, float scalar);
Matrix add_scalar(const Matrix& matrix, float value);

// --- elementwise math ---
Matrix exp(const Matrix& matrix);
Matrix log(const Matrix& matrix);
Matrix sqrt(const Matrix& matrix);
Matrix abs(const Matrix& matrix);
Matrix clamp(const Matrix& matrix, float low, float high);
Matrix sigmoid(const Matrix& matrix);
Matrix tanh(const Matrix& matrix);
Matrix leaky_relu(const Matrix& matrix, float negative_slope = 0.01f);
Matrix elu(const Matrix& matrix, float alpha = 1.0f);
Matrix gelu(const Matrix& matrix);
Matrix swish(const Matrix& matrix, float beta = 1.0f);
Matrix pow(const Matrix& matrix, float exponent);
Matrix negate(const Matrix& matrix);
Matrix divide(const Matrix& left, const Matrix& right);   // elementwise a / b
Matrix divide(const Matrix& matrix, float scalar);

} // namespace matrix_pro
