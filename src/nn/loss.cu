#include "matrix_pro/ops/operations.hpp"
#include "matrix_pro/core/cuda_utils.hpp"

#include <cuda_runtime.h>
#include <cfloat>
#include <stdexcept>

namespace matrix_pro {
namespace {

// Numerically stable elementwise BCE-with-logits:
// L = max(z,0) - z*t + log(1 + exp(-|z|))
__global__ void bce_with_logits_kernel(const float* logits, const float* target, float* loss, std::size_t count) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float z = logits[index];
    const float t = target[index];
    loss[index] = fmaxf(z, 0.0f) - z * t + logf(1.0f + expf(-fabsf(z)));
}

// dL/dz = (sigmoid(z) - t) / N   (for the mean-reduced loss)
__global__ void bce_with_logits_grad_kernel(const float* logits, const float* target, float* grad,
                                            std::size_t count, float scale) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float z = logits[index];
    const float sigmoid = 1.0f / (1.0f + expf(-z));
    grad[index] = scale * (sigmoid - target[index]);
}

// One block per row: fused softmax + cross-entropy with block reduction.
__global__ void softmax_cross_entropy_value_kernel(const float* logits, const float* target, float* loss,
                                                   std::size_t rows, std::size_t classes) {
    extern __shared__ float shared[];
    const std::size_t row = blockIdx.x;
    if (row >= rows) return;
    const float* x = logits + row * classes;
    const float* t = target + row * classes;

    float local_max = -FLT_MAX;
    for (std::size_t j = threadIdx.x; j < classes; j += blockDim.x) local_max = fmaxf(local_max, x[j]);
    shared[threadIdx.x] = local_max;
    __syncthreads();
    for (std::size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
        __syncthreads();
    }
    const float row_max = shared[0];

    float local_sum = 0.0f;
    for (std::size_t j = threadIdx.x; j < classes; j += blockDim.x) local_sum += expf(x[j] - row_max);
    shared[threadIdx.x] = local_sum;
    __syncthreads();
    for (std::size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    const float log_sum_exp = logf(shared[0]) + row_max;

    float local_dot = 0.0f;
    for (std::size_t j = threadIdx.x; j < classes; j += blockDim.x) local_dot += t[j] * x[j];
    shared[threadIdx.x] = local_dot;
    __syncthreads();
    for (std::size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) loss[row] = log_sum_exp - shared[0];
}

__global__ void softmax_cross_entropy_grad_kernel(const float* logits, const float* target, float* grad,
                                                  std::size_t rows, std::size_t classes, float scale) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto total = rows * classes;
    if (index >= total) return;
    const std::size_t row = index / classes;
    const float* x = logits + row * classes;
    float row_max = -FLT_MAX;
    for (std::size_t j = 0; j < classes; ++j) row_max = fmaxf(row_max, x[j]);
    float sum = 0.0f;
    for (std::size_t j = 0; j < classes; ++j) sum += expf(x[j] - row_max);
    const float softmax = expf(x[index % classes] - row_max) / sum;
    grad[index] = scale * (softmax - target[index]);
}

void validate_same(const Matrix& a, const Matrix& b, const char* message) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) throw InvalidArgumentError(message);
}

}

float bce_with_logits_value(const Matrix& logits, const Matrix& target) {
    validate_same(logits, target, "BCE logits/target shape mismatch");
    Matrix loss(logits.rows(), logits.cols());
    bce_with_logits_kernel<<<static_cast<unsigned>((loss.size() + 255) / 256), 256, 0, compute_stream()>>>(
        logits.device_data(), target.device_data(), loss.device_data(), loss.size());
    checkCuda(cudaGetLastError(), "bce with logits kernel launch");
    return sum(loss) / static_cast<float>(logits.size());
}

Matrix bce_with_logits_grad(const Matrix& logits, const Matrix& target) {
    validate_same(logits, target, "BCE logits/target shape mismatch");
    Matrix grad(logits.rows(), logits.cols());
    const float scale = 1.0f / static_cast<float>(logits.size());
    bce_with_logits_grad_kernel<<<static_cast<unsigned>((grad.size() + 255) / 256), 256, 0, compute_stream()>>>(
        logits.device_data(), target.device_data(), grad.device_data(), grad.size(), scale);
    checkCuda(cudaGetLastError(), "bce with logits grad kernel launch");
    grad.mark_host_stale();
    return grad;
}

Matrix softmax_cross_entropy_value(const Matrix& logits, const Matrix& target) {
    validate_same(logits, target, "Cross-entropy logits/target shape mismatch");
    const std::size_t rows = logits.rows(), classes = logits.cols();
    if (classes == 0) throw InvalidArgumentError("Cross-entropy requires at least one class");
    Matrix loss(rows, 1);
    if (rows != 0) {
        softmax_cross_entropy_value_kernel<<<static_cast<unsigned>(rows), 256,
            256 * sizeof(float), compute_stream()>>>(
            logits.device_data(), target.device_data(), loss.device_data(), rows, classes);
        checkCuda(cudaGetLastError(), "softmax cross entropy kernel launch");
    }
    loss.mark_host_stale();
    return loss;
}

Matrix softmax_cross_entropy_grad(const Matrix& logits, const Matrix& target) {
    validate_same(logits, target, "Cross-entropy logits/target shape mismatch");
    const std::size_t rows = logits.rows(), classes = logits.cols();
    Matrix grad(rows, classes);
    const float scale = 1.0f / static_cast<float>(rows);
    if (grad.size() != 0) {
        softmax_cross_entropy_grad_kernel<<<static_cast<unsigned>((grad.size() + 255) / 256), 256, 0, compute_stream()>>>(
            logits.device_data(), target.device_data(), grad.device_data(), rows, classes, scale);
        checkCuda(cudaGetLastError(), "softmax cross entropy grad kernel launch");
    }
    grad.mark_host_stale();
    return grad;
}

}
