#pragma once

#include <cstddef>
#include <string>

#include "matrix.hpp"

namespace matrix_pro {

// --- basic elementwise ---
Matrix add(const Matrix& left, const Matrix& right);
Matrix subtract(const Matrix& left, const Matrix& right);
Matrix elementwise_multiply(const Matrix& left, const Matrix& right);
Matrix multiply(const Matrix& left, const Matrix& right);   // matmul
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
Matrix pow(const Matrix& matrix, float exponent);
Matrix negate(const Matrix& matrix);

// --- transforms ---
Matrix transpose(const Matrix& matrix);
Matrix relu(const Matrix& matrix);
Matrix softmax(const Matrix& matrix);
Matrix flatten(const Matrix& matrix);
Matrix slice(const Matrix& matrix, std::size_t row_start, std::size_t row_end,
	std::size_t col_start, std::size_t col_end);

// --- product variants ---
Matrix outer_product(const Matrix& left, const Matrix& right);

// --- broadcast ---
Matrix add_row_vector(const Matrix& matrix, const Matrix& vector);     // v length == cols
Matrix add_col_vector(const Matrix& matrix, const Matrix& vector);     // v length == rows
Matrix multiply_row_vector(const Matrix& matrix, const Matrix& vector);
Matrix multiply_col_vector(const Matrix& matrix, const Matrix& vector);

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
float abs_max(const Matrix& matrix);
float trace(const Matrix& matrix);
float determinant(const Matrix& matrix);
Matrix inverse(const Matrix& matrix);

}

