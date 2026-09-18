#include "matrix_pro/ops/operations.hpp"
#include "matrix_pro/core/cuda_utils.hpp"

#include <cuda_runtime.h>
#include <stdexcept>

namespace matrix_pro {
namespace {

__global__ void softplus_kernel(const float* input, float* output, std::size_t count) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float x = input[index];
    // Overflow-safe softplus: log(1 + e^x) == x for x > 20.
    output[index] = x > 20.0f ? x : logf(1.0f + expf(x));
}

__global__ void softplus_backward_kernel(const float* input, const float* grad, float* output, std::size_t count) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float x = input[index];
    const float derivative = x > 20.0f ? 1.0f : 1.0f / (1.0f + expf(-x));
    output[index] = grad[index] * derivative;
}

__global__ void mish_kernel(const float* input, float* output, std::size_t count) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float x = input[index];
    const float sp = x > 20.0f ? x : logf(1.0f + expf(x));
    output[index] = x * tanhf(sp);
}

__global__ void mish_backward_kernel(const float* input, const float* grad, float* output, std::size_t count) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float x = input[index];
    const float sp = x > 20.0f ? x : logf(1.0f + expf(x));
    const float t = tanhf(sp);
    const float s = x > 20.0f ? 1.0f : 1.0f / (1.0f + expf(-x));
    // d/dx [x * tanh(sp)] = tanh(sp) + x * sech^2(sp) * sigmoid(x)
    const float sech2 = 1.0f - t * t;
    output[index] = grad[index] * (t + x * sech2 * s);
}

}

Matrix softplus(const Matrix& matrix) {
    Matrix output(matrix.rows(), matrix.cols());
    softplus_kernel<<<static_cast<unsigned>((output.size() + 255) / 256), 256, 0, compute_stream()>>>(
        matrix.device_data(), output.device_data(), output.size());
    checkCuda(cudaGetLastError(), "softplus kernel launch");
    output.mark_host_stale();
    return output;
}

Matrix softplus_backward(const Matrix& input, const Matrix& grad) {
    Matrix output(input.rows(), input.cols());
    softplus_backward_kernel<<<static_cast<unsigned>((output.size() + 255) / 256), 256, 0, compute_stream()>>>(
        input.device_data(), grad.device_data(), output.device_data(), output.size());
    checkCuda(cudaGetLastError(), "softplus backward kernel launch");
    output.mark_host_stale();
    return output;
}

Matrix mish(const Matrix& matrix) {
    Matrix output(matrix.rows(), matrix.cols());
    mish_kernel<<<static_cast<unsigned>((output.size() + 255) / 256), 256, 0, compute_stream()>>>(
        matrix.device_data(), output.device_data(), output.size());
    checkCuda(cudaGetLastError(), "mish kernel launch");
    output.mark_host_stale();
    return output;
}

Matrix mish_backward(const Matrix& input, const Matrix& grad) {
    Matrix output(input.rows(), input.cols());
    mish_backward_kernel<<<static_cast<unsigned>((output.size() + 255) / 256), 256, 0, compute_stream()>>>(
        input.device_data(), grad.device_data(), output.device_data(), output.size());
    checkCuda(cudaGetLastError(), "mish backward kernel launch");
    output.mark_host_stale();
    return output;
}

// The remaining activations reuse existing GPU primitives; keeping them in one
// place documents the exact formulas used by both Matrix and autograd layers.
Matrix hardtanh(const Matrix& matrix, float low, float high) { return clamp(matrix, low, high); }
Matrix hardsigmoid(const Matrix& matrix) { return clamp(matrix * (1.0f / 6.0f), -0.5f, 0.5f).add_scalar(0.5f); }
// Note: `matrix * other` is the GEMM operator, so the elementwise product must
// go through elementwise_multiply here.
Matrix hardswish(const Matrix& matrix) { return matrix_pro::elementwise_multiply(matrix, hardsigmoid(matrix)); }

Matrix selu(const Matrix& matrix) {
    const float lambda = 1.0507009873554805f;   // SELU_LAMBDA
    const float alpha = 1.6732632423543772f;    // SELU_ALPHA
    Matrix positive = matrix * lambda;
    // SELU negative branch: lambda * alpha * (exp(x) - 1)
    // NOT lambda * alpha * (exp(x) - alpha)
    Matrix negative = matrix.exp().add_scalar(-1.0f) * (lambda * alpha);
    return where(matrix.greater(0.0f), positive, negative);
}

Matrix prelu(const Matrix& matrix, const Matrix& alpha) {
    const bool scalar = alpha.rows() == 1 && alpha.cols() == 1;
    const bool exact = alpha.rows() == matrix.rows() && alpha.cols() == matrix.cols();
    const bool per_column = alpha.rows() == 1 && alpha.cols() == matrix.cols();
    if (!scalar && !exact && !per_column)
        throw ShapeMismatchError("PReLU alpha must be 1x1, 1xC or match input shape");
    Matrix scaled = broadcast_multiply(matrix, alpha);
    return where(matrix.greater(0.0f), matrix, scaled);
}

}
