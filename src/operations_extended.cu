#include "matrix_pro/operations.hpp"
#include "matrix_pro/cuda_utils.hpp"

#include <stdexcept>

namespace matrix_pro {
namespace {
__global__ void kron_kernel(const float* left, const float* right, float* output,
                            std::size_t left_rows, std::size_t left_cols,
                            std::size_t right_rows, std::size_t right_cols) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = left_rows * right_rows * left_cols * right_cols;
    if (index >= total) return;
    const std::size_t output_cols = left_cols * right_cols;
    const std::size_t row = index / output_cols;
    const std::size_t col = index % output_cols;
    const std::size_t left_row = row / right_rows;
    const std::size_t right_row = row % right_rows;
    const std::size_t left_col = col / right_cols;
    const std::size_t right_col = col % right_cols;
    output[index] = left[left_row * left_cols + left_col] * right[right_row * right_cols + right_col];
}
}

Matrix kron(const Matrix& left, const Matrix& right) {
    Matrix output(left.rows() * right.rows(), left.cols() * right.cols());
    if (output.empty()) return output;
    kron_kernel<<<static_cast<unsigned>((output.size() + 255) / 256), 256, 0, compute_stream()>>>(
        left.device_data(), right.device_data(), output.device_data(),
        left.rows(), left.cols(), right.rows(), right.cols());
    checkCuda(cudaGetLastError(), "kron kernel launch");
    return output;
}

Matrix& Matrix::operator+=(const Matrix& other) {
    *this = matrix_pro::add(*this, other);
    return *this;
}

Matrix& Matrix::operator-=(const Matrix& other) {
    *this = matrix_pro::subtract(*this, other);
    return *this;
}

Matrix& Matrix::operator*=(float scalar) {
    *this = matrix_pro::multiply(*this, scalar);
    return *this;
}

Matrix& Matrix::operator/=(float scalar) {
    if (scalar == 0.0f) throw std::invalid_argument("Matrix division by zero");
    *this = matrix_pro::multiply(*this, 1.0f / scalar);
    return *this;
}

Matrix Matrix::kron(const Matrix& other) const { return matrix_pro::kron(*this, other); }
}