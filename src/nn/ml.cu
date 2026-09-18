#include "matrix_pro/ops/operations.hpp"
#include "matrix_pro/core/cuda_utils.hpp"

#include <cmath>
#include <stdexcept>

namespace matrix_pro {
namespace {

enum class Activation { leaky_relu, elu, gelu, swish };

__device__ float activation_value(float value, Activation activation, float parameter) {
    if (activation == Activation::leaky_relu) return value > 0.0f ? value : parameter * value;
    if (activation == Activation::elu) return value > 0.0f ? value : parameter * (expf(value) - 1.0f);
    if (activation == Activation::gelu) {
        const float x3 = value * value * value;
        return 0.5f * value * (1.0f + tanhf(0.7978845608f * (value + 0.044715f * x3)));
    }
    const float sigmoid = 1.0f / (1.0f + expf(-parameter * value));
    return value * sigmoid;
}

__global__ void activation_kernel(const float* input, float* output, std::size_t count,
                                  Activation activation, float parameter) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = activation_value(input[index], activation, parameter);
}

__global__ void batch_norm_kernel(const float* input, float* output,
                                  std::size_t rows, std::size_t cols, float epsilon) {
    const auto col = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (col >= cols) return;
    float mean = 0.0f;
    for (std::size_t row = 0; row < rows; ++row) mean += input[row * cols + col];
    mean /= static_cast<float>(rows);
    float variance = 0.0f;
    for (std::size_t row = 0; row < rows; ++row) {
        const float delta = input[row * cols + col] - mean;
        variance += delta * delta;
    }
    variance /= static_cast<float>(rows);
    const float inverse_std = rsqrtf(variance + epsilon);
    for (std::size_t row = 0; row < rows; ++row) output[row * cols + col] = (input[row * cols + col] - mean) * inverse_std;
}

__global__ void layer_norm_kernel(const float* input, float* output,
                                  std::size_t rows, std::size_t cols, float epsilon) {
    const auto row = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    float mean = 0.0f;
    for (std::size_t col = 0; col < cols; ++col) mean += input[row * cols + col];
    mean /= static_cast<float>(cols);
    float variance = 0.0f;
    for (std::size_t col = 0; col < cols; ++col) {
        const float delta = input[row * cols + col] - mean;
        variance += delta * delta;
    }
    variance /= static_cast<float>(cols);
    const float inverse_std = rsqrtf(variance + epsilon);
    for (std::size_t col = 0; col < cols; ++col) output[row * cols + col] = (input[row * cols + col] - mean) * inverse_std;
}

__device__ unsigned int xorshift32(unsigned int value) {
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    return value;
}

__global__ void dropout_kernel(const float* input, float* output, std::size_t count,
                               float probability, unsigned long long seed) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const unsigned int mixed_seed = static_cast<unsigned int>(seed) ^ static_cast<unsigned int>(index * 747796405ULL + 2891336453ULL);
    const unsigned int random_bits = xorshift32(mixed_seed == 0 ? 1U : mixed_seed);
    const float random_value = static_cast<float>(random_bits) / 4294967296.0f;
    output[index] = random_value < probability ? 0.0f : input[index] / (1.0f - probability);
}

Matrix activation(const Matrix& matrix, Activation operation, float parameter) {
    Matrix output(matrix.rows(), matrix.cols());
    if (matrix.empty()) return output;
    activation_kernel<<<static_cast<unsigned>((matrix.size() + 255) / 256), 256, 0, compute_stream()>>>(
        matrix.device_data(), output.device_data(), matrix.size(), operation, parameter);
    checkCuda(cudaGetLastError(), "activation kernel launch");
    output.mark_host_stale();
    return output;
}
}

Matrix leaky_relu(const Matrix& matrix, float negative_slope) {
    return activation(matrix, Activation::leaky_relu, negative_slope);
}

Matrix elu(const Matrix& matrix, float alpha) {
    if (alpha < 0.0f) throw InvalidArgumentError("ELU alpha must be non-negative");
    return activation(matrix, Activation::elu, alpha);
}

Matrix gelu(const Matrix& matrix) { return activation(matrix, Activation::gelu, 0.0f); }
Matrix swish(const Matrix& matrix, float beta) { return activation(matrix, Activation::swish, beta); }

Matrix batch_norm(const Matrix& matrix, float epsilon) {
    if (matrix.empty()) return Matrix(matrix.rows(), matrix.cols());
    if (epsilon <= 0.0f) throw InvalidArgumentError("Batch norm epsilon must be positive");
    Matrix output(matrix.rows(), matrix.cols());
    batch_norm_kernel<<<static_cast<unsigned>((matrix.cols() + 255) / 256), 256, 0, compute_stream()>>>(
        matrix.device_data(), output.device_data(), matrix.rows(), matrix.cols(), epsilon);
    checkCuda(cudaGetLastError(), "batch norm kernel launch");
    output.mark_host_stale();
    return output;
}

Matrix layer_norm(const Matrix& matrix, float epsilon) {
    if (matrix.empty()) return Matrix(matrix.rows(), matrix.cols());
    if (epsilon <= 0.0f) throw InvalidArgumentError("Layer norm epsilon must be positive");
    Matrix output(matrix.rows(), matrix.cols());
    layer_norm_kernel<<<static_cast<unsigned>((matrix.rows() + 255) / 256), 256, 0, compute_stream()>>>(
        matrix.device_data(), output.device_data(), matrix.rows(), matrix.cols(), epsilon);
    checkCuda(cudaGetLastError(), "layer norm kernel launch");
    output.mark_host_stale();
    return output;
}

Matrix dropout(const Matrix& matrix, float probability, unsigned long long seed) {
    if (probability < 0.0f || probability >= 1.0f) throw InvalidArgumentError("Dropout probability must be in [0, 1)");
    if (probability == 0.0f) return matrix;
    Matrix output(matrix.rows(), matrix.cols());
    dropout_kernel<<<static_cast<unsigned>((matrix.size() + 255) / 256), 256, 0, compute_stream()>>>(
        matrix.device_data(), output.device_data(), matrix.size(), probability, seed);
    checkCuda(cudaGetLastError(), "dropout kernel launch");
    output.mark_host_stale();
    return output;
}

Matrix Matrix::leaky_relu(float negative_slope) const { return matrix_pro::leaky_relu(*this, negative_slope); }
Matrix Matrix::elu(float alpha) const { return matrix_pro::elu(*this, alpha); }
Matrix Matrix::gelu() const { return matrix_pro::gelu(*this); }
Matrix Matrix::swish(float beta) const { return matrix_pro::swish(*this, beta); }
}
