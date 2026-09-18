#include "matrix_pro/ops/operations.hpp"
#include "matrix_pro/core/cuda_utils.hpp"
#include "matrix_pro/detail/kernel_helpers.cuh"

#include <stdexcept>

namespace matrix_pro {
namespace {

enum class UnaryOp {
    exp, log, sqrt, abs, clamp, sigmoid, tanh, pow_scalar, negate
};

// Elementwise functors for the vectorized launch helpers. Keeping the work in
// functors lets a single templated kernel read/write 16 bytes per thread while
// still serving every operation.
struct AddOp { __device__ float operator()(float a, float b) const { return a + b; } };
struct SubtractOp { __device__ float operator()(float a, float b) const { return a - b; } };
struct MultiplyOp { __device__ float operator()(float a, float b) const { return a * b; } };
struct MultiplyScalarOp { float scalar; __device__ float operator()(float a) const { return a * scalar; } };
struct AddScalarOp { float scalar; __device__ float operator()(float a) const { return a + scalar; } };

struct UnaryDispatcher {
    UnaryOp op;
    float param_a;
    float param_b;
    __device__ float operator()(float value) const {
        switch (op) {
            case UnaryOp::exp:        return expf(value);
            case UnaryOp::log:        return logf(value);
            case UnaryOp::sqrt:       return sqrtf(value);
            case UnaryOp::abs:        return fabsf(value);
            case UnaryOp::clamp:      return fminf(fmaxf(value, param_a), param_b);
            case UnaryOp::sigmoid:    return 1.0f / (1.0f + expf(-value));
            case UnaryOp::tanh:       return tanhf(value);
            case UnaryOp::pow_scalar: return powf(value, param_a);
            case UnaryOp::negate:     return -value;
        }
        return value;
    }
};

__global__ void add_row_vector_kernel(const float* __restrict__ matrix, const float* __restrict__ vector,
                                      float* __restrict__ output, std::size_t rows, std::size_t cols) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rows * cols) return;
    output[index] = matrix[index] + vector[index % cols];
}

__global__ void add_col_vector_kernel(const float* __restrict__ matrix, const float* __restrict__ vector,
                                      float* __restrict__ output, std::size_t rows, std::size_t cols) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rows * cols) return;
    output[index] = matrix[index] + vector[index / cols];
}

__global__ void mul_row_vector_kernel(const float* __restrict__ matrix, const float* __restrict__ vector,
                                      float* __restrict__ output, std::size_t rows, std::size_t cols) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rows * cols) return;
    output[index] = matrix[index] * vector[index % cols];
}

__global__ void mul_col_vector_kernel(const float* __restrict__ matrix, const float* __restrict__ vector,
                                      float* __restrict__ output, std::size_t rows, std::size_t cols) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rows * cols) return;
    output[index] = matrix[index] * vector[index / cols];
}

Matrix unary(const Matrix& matrix, UnaryOp op, float a = 0.0f, float b = 0.0f) {
    Matrix output(matrix.rows(), matrix.cols());
    detail::launch_unary(matrix.device_data(), output.device_data(), matrix.size(),
                         UnaryDispatcher{op, a, b}, compute_stream());
    checkCuda(cudaGetLastError(), "unary kernel launch");
    output.mark_host_stale();
    return output;
}

}

Matrix add(const Matrix& left, const Matrix& right) {
    if (left.rows() != right.rows() || left.cols() != right.cols()) {
        throw ShapeMismatchError("Matrix shapes must match");
    }
    Matrix output(left.rows(), left.cols());
    detail::launch_binary(left.device_data(), right.device_data(), output.device_data(),
                          left.size(), AddOp{}, compute_stream());
    checkCuda(cudaGetLastError(), "add kernel launch");
    output.mark_host_stale();
    return output;
}

Matrix subtract(const Matrix& left, const Matrix& right) {
    if (left.rows() != right.rows() || left.cols() != right.cols()) {
        throw ShapeMismatchError("Matrix shapes must match");
    }
    Matrix output(left.rows(), left.cols());
    detail::launch_binary(left.device_data(), right.device_data(), output.device_data(),
                          left.size(), SubtractOp{}, compute_stream());
    checkCuda(cudaGetLastError(), "subtract kernel launch");
    output.mark_host_stale();
    return output;
}

Matrix elementwise_multiply(const Matrix& left, const Matrix& right) {
    if (left.rows() != right.rows() || left.cols() != right.cols()) {
        throw ShapeMismatchError("Matrix shapes must match");
    }
    Matrix output(left.rows(), left.cols());
    detail::launch_binary(left.device_data(), right.device_data(), output.device_data(),
                          left.size(), MultiplyOp{}, compute_stream());
    checkCuda(cudaGetLastError(), "elementwise multiply kernel launch");
    output.mark_host_stale();
    return output;
}

Matrix multiply(const Matrix& matrix, float scalar) {
    Matrix output(matrix.rows(), matrix.cols());
    detail::launch_unary(matrix.device_data(), output.device_data(), matrix.size(),
                         MultiplyScalarOp{scalar}, compute_stream());
    checkCuda(cudaGetLastError(), "scalar kernel launch");
    output.mark_host_stale();
    return output;
}

Matrix add_scalar(const Matrix& matrix, float value) {
    Matrix output(matrix.rows(), matrix.cols());
    detail::launch_unary(matrix.device_data(), output.device_data(), matrix.size(),
                         AddScalarOp{value}, compute_stream());
    checkCuda(cudaGetLastError(), "add scalar kernel launch");
    output.mark_host_stale();
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
    if (vector.size() != matrix.cols()) throw ShapeMismatchError("Row vector length must equal column count");
    Matrix output(matrix.rows(), matrix.cols());
    add_row_vector_kernel<<<static_cast<unsigned>((matrix.size() + 255) / 256), 256, 0, compute_stream()>>>(
        matrix.device_data(), vector.device_data(), output.device_data(), matrix.rows(), matrix.cols());
    checkCuda(cudaGetLastError(), "add row vector kernel launch");
    output.mark_host_stale();
    return output;
}

Matrix add_col_vector(const Matrix& matrix, const Matrix& vector) {
    if (vector.size() != matrix.rows()) throw ShapeMismatchError("Column vector length must equal row count");
    Matrix output(matrix.rows(), matrix.cols());
    add_col_vector_kernel<<<static_cast<unsigned>((matrix.size() + 255) / 256), 256, 0, compute_stream()>>>(
        matrix.device_data(), vector.device_data(), output.device_data(), matrix.rows(), matrix.cols());
    checkCuda(cudaGetLastError(), "add col vector kernel launch");
    output.mark_host_stale();
    return output;
}

Matrix multiply_row_vector(const Matrix& matrix, const Matrix& vector) {
    if (vector.size() != matrix.cols()) throw ShapeMismatchError("Row vector length must equal column count");
    Matrix output(matrix.rows(), matrix.cols());
    mul_row_vector_kernel<<<static_cast<unsigned>((matrix.size() + 255) / 256), 256, 0, compute_stream()>>>(
        matrix.device_data(), vector.device_data(), output.device_data(), matrix.rows(), matrix.cols());
    checkCuda(cudaGetLastError(), "multiply row vector kernel launch");
    output.mark_host_stale();
    return output;
}

Matrix multiply_col_vector(const Matrix& matrix, const Matrix& vector) {
    if (vector.size() != matrix.rows()) throw ShapeMismatchError("Column vector length must equal row count");
    Matrix output(matrix.rows(), matrix.cols());
    mul_col_vector_kernel<<<static_cast<unsigned>((matrix.size() + 255) / 256), 256, 0, compute_stream()>>>(
        matrix.device_data(), vector.device_data(), output.device_data(), matrix.rows(), matrix.cols());
    checkCuda(cudaGetLastError(), "multiply col vector kernel launch");
    output.mark_host_stale();
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

