#include "matrix_pro/operations.hpp"
#include "matrix_pro/cuda_utils.hpp"

#include <stdexcept>

namespace matrix_pro {
namespace {

enum class BinaryOperation { add, subtract, multiply };

__global__ void binary_kernel(const float* left, const float* right, float* output,
                              std::size_t count, BinaryOperation operation) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    if (operation == BinaryOperation::add) output[index] = left[index] + right[index];
    if (operation == BinaryOperation::subtract) output[index] = left[index] - right[index];
    if (operation == BinaryOperation::multiply) output[index] = left[index] * right[index];
}

__global__ void scalar_kernel(const float* input, float* output, std::size_t count, float scalar) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) output[index] = input[index] * scalar;
}

Matrix binary_operation(const Matrix& left, const Matrix& right, BinaryOperation operation) {
    if (left.rows() != right.rows() || left.cols() != right.cols()) {
        throw std::invalid_argument("Matrix shapes must match");
    }
    Matrix output(left.rows(), left.cols());
    binary_kernel<<<static_cast<unsigned>((left.size() + 255) / 256), 256>>>(
        left.device_data(), right.device_data(), output.device_data(), left.size(), operation);
    checkCuda(cudaGetLastError(), "binary kernel launch");
    return output;
}

}

Matrix add(const Matrix& left, const Matrix& right) { return binary_operation(left, right, BinaryOperation::add); }
Matrix subtract(const Matrix& left, const Matrix& right) { return binary_operation(left, right, BinaryOperation::subtract); }
Matrix elementwise_multiply(const Matrix& left, const Matrix& right) { return binary_operation(left, right, BinaryOperation::multiply); }

Matrix multiply(const Matrix& matrix, float scalar) {
    Matrix output(matrix.rows(), matrix.cols());
    scalar_kernel<<<static_cast<unsigned>((matrix.size() + 255) / 256), 256>>>(
        matrix.device_data(), output.device_data(), matrix.size(), scalar);
    checkCuda(cudaGetLastError(), "scalar kernel launch");
    return output;
}

Matrix operator*(float scalar, const Matrix& matrix) { return matrix * scalar; }
Matrix Matrix::operator+(const Matrix& other) const { return add(*this, other); }
Matrix Matrix::operator-(const Matrix& other) const { return subtract(*this, other); }
Matrix Matrix::operator*(const Matrix& other) const { return multiply(*this, other); }
Matrix Matrix::operator*(float scalar) const { return multiply(*this, scalar); }
Matrix Matrix::elementwise_multiply(const Matrix& other) const { return matrix_pro::elementwise_multiply(*this, other); }

}
