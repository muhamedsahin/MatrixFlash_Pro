#include "matrix_pro/operations.hpp"
#include "matrix_pro/cuda_utils.hpp"

#include <stdexcept>

namespace matrix_pro {
namespace {

enum class BinaryOperation { add, subtract, multiply };

enum class UnaryOp {
    exp, log, sqrt, abs, clamp, sigmoid, tanh, pow_scalar, negate
};

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

__global__ void bias_kernel(const float* input, float* output, std::size_t count, float bias) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) output[index] = input[index] + bias;
}

__global__ void unary_kernel(const float* input, float* output, std::size_t count,
                             UnaryOp op, float param_a, float param_b) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float value = input[index];
    float result;
    switch (op) {
        case UnaryOp::exp:        result = expf(value); break;
        case UnaryOp::log:        result = logf(value); break;
        case UnaryOp::sqrt:       result = sqrtf(value); break;
        case UnaryOp::abs:        result = fabsf(value); break;
        case UnaryOp::clamp:      result = fminf(fmaxf(value, param_a), param_b); break;
        case UnaryOp::sigmoid:    result = 1.0f / (1.0f + expf(-value)); break;
        case UnaryOp::tanh:       result = tanhf(value); break;
        case UnaryOp::pow_scalar: result = powf(value, param_a); break;
        case UnaryOp::negate:     result = -value; break;
        default:                  result = value; break;
    }
    output[index] = result;
}

__global__ void add_row_vector_kernel(const float* matrix, const float* vector, float* output,
                                      std::size_t rows, std::size_t cols) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rows * cols) return;
    output[index] = matrix[index] + vector[index % cols];
}

__global__ void add_col_vector_kernel(const float* matrix, const float* vector, float* output,
                                      std::size_t rows, std::size_t cols) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rows * cols) return;
    output[index] = matrix[index] + vector[index / cols];
}

__global__ void mul_row_vector_kernel(const float* matrix, const float* vector, float* output,
                                      std::size_t rows, std::size_t cols) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rows * cols) return;
    output[index] = matrix[index] * vector[index % cols];
}

__global__ void mul_col_vector_kernel(const float* matrix, const float* vector, float* output,
                                      std::size_t rows, std::size_t cols) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rows * cols) return;
    output[index] = matrix[index] * vector[index / cols];
}

Matrix unary(const Matrix& matrix, UnaryOp op, float a = 0.0f, float b = 0.0f) {
    Matrix output(matrix.rows(), matrix.cols());
    unary_kernel<<<static_cast<unsigned>((matrix.size() + 255) / 256), 256>>>(
        matrix.device_data(), output.device_data(), matrix.size(), op, a, b);
    checkCuda(cudaGetLastError(), "unary kernel launch");
    return output;
}

}

Matrix add(const Matrix& left, const Matrix& right) {
    if (left.rows() != right.rows() || left.cols() != right.cols()) {
        throw std::invalid_argument("Matrix shapes must match");
    }
    Matrix output(left.rows(), left.cols());
    binary_kernel<<<static_cast<unsigned>((left.size() + 255) / 256), 256>>>(
        left.device_data(), right.device_data(), output.device_data(), left.size(), BinaryOperation::add);
    checkCuda(cudaGetLastError(), "add kernel launch");
    return output;
}

Matrix subtract(const Matrix& left, const Matrix& right) {
    if (left.rows() != right.rows() || left.cols() != right.cols()) {
        throw std::invalid_argument("Matrix shapes must match");
    }
    Matrix output(left.rows(), left.cols());
    binary_kernel<<<static_cast<unsigned>((left.size() + 255) / 256), 256>>>(
        left.device_data(), right.device_data(), output.device_data(), left.size(), BinaryOperation::subtract);
    checkCuda(cudaGetLastError(), "subtract kernel launch");
    return output;
}

Matrix elementwise_multiply(const Matrix& left, const Matrix& right) {
    if (left.rows() != right.rows() || left.cols() != right.cols()) {
        throw std::invalid_argument("Matrix shapes must match");
    }
    Matrix output(left.rows(), left.cols());
    binary_kernel<<<static_cast<unsigned>((left.size() + 255) / 256), 256>>>(
        left.device_data(), right.device_data(), output.device_data(), left.size(), BinaryOperation::multiply);
    checkCuda(cudaGetLastError(), "elementwise multiply kernel launch");
    return output;
}

Matrix multiply(const Matrix& matrix, float scalar) {
    Matrix output(matrix.rows(), matrix.cols());
    scalar_kernel<<<static_cast<unsigned>((matrix.size() + 255) / 256), 256>>>(
        matrix.device_data(), output.device_data(), matrix.size(), scalar);
    checkCuda(cudaGetLastError(), "scalar kernel launch");
    return output;
}

Matrix add_scalar(const Matrix& matrix, float value) {
    Matrix output(matrix.rows(), matrix.cols());
    bias_kernel<<<static_cast<unsigned>((matrix.size() + 255) / 256), 256>>>(
        matrix.device_data(), output.device_data(), matrix.size(), value);
    checkCuda(cudaGetLastError(), "add scalar kernel launch");
    return output;
}

Matrix exp(const Matrix& matrix) { return unary(matrix, UnaryOp::exp); }
Matrix log(const Matrix& matrix) { return unary(matrix, UnaryOp::log); }
Matrix sqrt(const Matrix& matrix) { return unary(matrix, UnaryOp::sqrt); }
Matrix abs(const Matrix& matrix) { return unary(matrix, UnaryOp::abs); }
Matrix clamp(const Matrix& matrix, float low, float high) { return unary(matrix, UnaryOp::clamp, low, high); }
Matrix sigmoid(const Matrix& matrix) { return unary(matrix, UnaryOp::sigmoid); }
Matrix tanh(const Matrix& matrix) { return unary(matrix, UnaryOp::tanh); }
Matrix pow(const Matrix& matrix, float exponent) { return unary(matrix, UnaryOp::pow_scalar, exponent); }
Matrix negate(const Matrix& matrix) { return unary(matrix, UnaryOp::negate); }

Matrix add_row_vector(const Matrix& matrix, const Matrix& vector) {
    if (vector.size() != matrix.cols()) throw std::invalid_argument("Row vector length must equal column count");
    Matrix output(matrix.rows(), matrix.cols());
    add_row_vector_kernel<<<static_cast<unsigned>((matrix.size() + 255) / 256), 256>>>(
        matrix.device_data(), vector.device_data(), output.device_data(), matrix.rows(), matrix.cols());
    checkCuda(cudaGetLastError(), "add row vector kernel launch");
    return output;
}

Matrix add_col_vector(const Matrix& matrix, const Matrix& vector) {
    if (vector.size() != matrix.rows()) throw std::invalid_argument("Column vector length must equal row count");
    Matrix output(matrix.rows(), matrix.cols());
    add_col_vector_kernel<<<static_cast<unsigned>((matrix.size() + 255) / 256), 256>>>(
        matrix.device_data(), vector.device_data(), output.device_data(), matrix.rows(), matrix.cols());
    checkCuda(cudaGetLastError(), "add col vector kernel launch");
    return output;
}

Matrix multiply_row_vector(const Matrix& matrix, const Matrix& vector) {
    if (vector.size() != matrix.cols()) throw std::invalid_argument("Row vector length must equal column count");
    Matrix output(matrix.rows(), matrix.cols());
    mul_row_vector_kernel<<<static_cast<unsigned>((matrix.size() + 255) / 256), 256>>>(
        matrix.device_data(), vector.device_data(), output.device_data(), matrix.rows(), matrix.cols());
    checkCuda(cudaGetLastError(), "multiply row vector kernel launch");
    return output;
}

Matrix multiply_col_vector(const Matrix& matrix, const Matrix& vector) {
    if (vector.size() != matrix.rows()) throw std::invalid_argument("Column vector length must equal row count");
    Matrix output(matrix.rows(), matrix.cols());
    mul_col_vector_kernel<<<static_cast<unsigned>((matrix.size() + 255) / 256), 256>>>(
        matrix.device_data(), vector.device_data(), output.device_data(), matrix.rows(), matrix.cols());
    checkCuda(cudaGetLastError(), "multiply col vector kernel launch");
    return output;
}

Matrix operator*(float scalar, const Matrix& matrix) { return matrix * scalar; }

// --- Matrix method wrappers ---
Matrix Matrix::operator+() const { return *this; }
Matrix Matrix::operator-() const { return matrix_pro::negate(*this); }
Matrix Matrix::operator+(const Matrix& other) const { return matrix_pro::add(*this, other); }
Matrix Matrix::operator-(const Matrix& other) const { return matrix_pro::subtract(*this, other); }
Matrix Matrix::operator*(float scalar) const { return matrix_pro::multiply(*this, scalar); }
Matrix Matrix::elementwise_multiply(const Matrix& other) const { return matrix_pro::elementwise_multiply(*this, other); }
Matrix Matrix::add_scalar(float value) const { return matrix_pro::add_scalar(*this, value); }

Matrix Matrix::exp() const { return matrix_pro::exp(*this); }
Matrix Matrix::log() const { return matrix_pro::log(*this); }
Matrix Matrix::sqrt() const { return matrix_pro::sqrt(*this); }
Matrix Matrix::abs() const { return matrix_pro::abs(*this); }
Matrix Matrix::clamp(float low, float high) const { return matrix_pro::clamp(*this, low, high); }
Matrix Matrix::sigmoid() const { return matrix_pro::sigmoid(*this); }
Matrix Matrix::tanh() const { return matrix_pro::tanh(*this); }
Matrix Matrix::pow(float exponent) const { return matrix_pro::pow(*this, exponent); }

Matrix Matrix::add_row_vector(const Matrix& v) const { return matrix_pro::add_row_vector(*this, v); }
Matrix Matrix::add_col_vector(const Matrix& v) const { return matrix_pro::add_col_vector(*this, v); }
Matrix Matrix::multiply_row_vector(const Matrix& v) const { return matrix_pro::multiply_row_vector(*this, v); }
Matrix Matrix::multiply_col_vector(const Matrix& v) const { return matrix_pro::multiply_col_vector(*this, v); }

}

