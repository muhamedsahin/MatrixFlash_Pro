#include "matrix_pro/operations.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace matrix_pro {

float determinant(const Matrix& matrix) {
    if (matrix.rows() != matrix.cols()) throw std::invalid_argument("Determinant requires a square matrix");
    Matrix work = matrix;
    work.download();
    const std::size_t n = matrix.rows();
    float result = 1.0f;
    for (std::size_t pivot = 0; pivot < n; ++pivot) {
        std::size_t row = pivot;
        for (std::size_t candidate = pivot + 1; candidate < n; ++candidate) {
            if (std::fabs(work.at(candidate, pivot)) > std::fabs(work.at(row, pivot))) row = candidate;
        }
        if (std::fabs(work.at(row, pivot)) < 1e-7f) return 0.0f;
        if (row != pivot) {
            for (std::size_t col = 0; col < n; ++col) std::swap(work.at(row, col), work.at(pivot, col));
            result = -result;
        }
        const float diagonal = work.at(pivot, pivot);
        result *= diagonal;
        for (std::size_t candidate = pivot + 1; candidate < n; ++candidate) {
            const float factor = work.at(candidate, pivot) / diagonal;
            for (std::size_t col = pivot + 1; col < n; ++col) work.at(candidate, col) -= factor * work.at(pivot, col);
        }
    }
    return result;
}

Matrix inverse(const Matrix& matrix) {
    if (matrix.rows() != matrix.cols()) throw std::invalid_argument("Inverse requires a square matrix");
    const std::size_t n = matrix.rows();
    Matrix work = matrix;
    work.download();
    Matrix result = Matrix::identity(n);
    result.download();
    for (std::size_t pivot = 0; pivot < n; ++pivot) {
        std::size_t row = pivot;
        for (std::size_t candidate = pivot + 1; candidate < n; ++candidate) {
            if (std::fabs(work.at(candidate, pivot)) > std::fabs(work.at(row, pivot))) row = candidate;
        }
        if (std::fabs(work.at(row, pivot)) < 1e-7f) throw std::runtime_error("Matrix is singular");
        for (std::size_t col = 0; col < n; ++col) {
            std::swap(work.at(row, col), work.at(pivot, col));
            std::swap(result.at(row, col), result.at(pivot, col));
        }
        const float diagonal = work.at(pivot, pivot);
        for (std::size_t col = 0; col < n; ++col) {
            work.at(pivot, col) /= diagonal;
            result.at(pivot, col) /= diagonal;
        }
        for (std::size_t candidate = 0; candidate < n; ++candidate) {
            if (candidate == pivot) continue;
            const float factor = work.at(candidate, pivot);
            for (std::size_t col = 0; col < n; ++col) {
                work.at(candidate, col) -= factor * work.at(pivot, col);
                result.at(candidate, col) -= factor * result.at(pivot, col);
            }
        }
    }
    result.upload();
    return result;
}

float Matrix::determinant() const { return matrix_pro::determinant(*this); }
Matrix Matrix::inverse() const { return matrix_pro::inverse(*this); }

}
