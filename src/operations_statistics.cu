#include "matrix_pro/operations.hpp"

#include <cmath>
#include <numeric>
#include <stdexcept>

namespace matrix_pro {

float sum(const Matrix& matrix) {
    Matrix copy = matrix;
    copy.download();
    return std::accumulate(copy.data().begin(), copy.data().end(), 0.0f);
}

float mean(const Matrix& matrix) {
    if (matrix.empty()) throw std::invalid_argument("Mean is undefined for an empty matrix");
    return sum(matrix) / static_cast<float>(matrix.size());
}

float l2_norm(const Matrix& matrix) {
    Matrix copy = matrix;
    copy.download();
    float result = 0.0f;
    for (const float value : copy.data()) result += value * value;
    return std::sqrt(result);
}

float trace(const Matrix& matrix) {
    if (matrix.rows() != matrix.cols()) throw std::invalid_argument("Trace requires a square matrix");
    Matrix copy = matrix;
    copy.download();
    float result = 0.0f;
    for (std::size_t index = 0; index < matrix.rows(); ++index) result += copy.at(index, index);
    return result;
}

float Matrix::sum() const { return matrix_pro::sum(*this); }
float Matrix::mean() const { return matrix_pro::mean(*this); }
float Matrix::l2_norm() const { return matrix_pro::l2_norm(*this); }
float Matrix::trace() const { return matrix_pro::trace(*this); }

}
