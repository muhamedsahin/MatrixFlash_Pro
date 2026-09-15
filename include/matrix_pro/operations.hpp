#pragma once

#include "matrix.hpp"

namespace matrix_pro {

Matrix add(const Matrix& left, const Matrix& right);
Matrix subtract(const Matrix& left, const Matrix& right);
Matrix multiply(const Matrix& left, const Matrix& right);
Matrix elementwise_multiply(const Matrix& left, const Matrix& right);
Matrix multiply(const Matrix& matrix, float scalar);
Matrix transpose(const Matrix& matrix);
Matrix relu(const Matrix& matrix);
Matrix softmax(const Matrix& matrix);
Matrix flatten(const Matrix& matrix);
Matrix slice(const Matrix& matrix, std::size_t row_start, std::size_t row_end,
	std::size_t col_start, std::size_t col_end);
float sum(const Matrix& matrix);
float mean(const Matrix& matrix);
float l2_norm(const Matrix& matrix);
float trace(const Matrix& matrix);
float determinant(const Matrix& matrix);
Matrix inverse(const Matrix& matrix);

}
