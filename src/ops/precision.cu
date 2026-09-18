#include "matrix_pro/ops/operations.hpp"
#include "matrix_pro/core/cuda_utils.hpp"

#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <stdexcept>

namespace matrix_pro {
namespace {

__global__ void float_to_half_kernel(const float* input, __half* output, std::size_t count) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = __float2half(input[index]);
}

__global__ void double_accumulate_kernel(const float* left, const float* right, float* output,
                                         std::size_t rows, std::size_t inner, std::size_t cols) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto total = rows * cols;
    if (index >= total) return;
    const auto row = index / cols;
    const auto col = index % cols;
    double sum = 0.0;
    for (std::size_t i = 0; i < inner; ++i) {
        sum += static_cast<double>(left[row * inner + i]) * static_cast<double>(right[i * cols + col]);
    }
    output[index] = static_cast<float>(sum);
}

}

Matrix matmul_half(const Matrix& left, const Matrix& right) {
    if (left.cols() != right.rows()) throw ShapeMismatchError("Matrix dimensions are incompatible for matmul_half");
    Matrix output(left.rows(), right.cols());
    if (output.empty()) return output;
    const auto left_bytes = left.size() * sizeof(__half);
    const auto right_bytes = right.size() * sizeof(__half);
    auto* left_half = static_cast<__half*>(allocate_device_memory(left_bytes));
    auto* right_half = static_cast<__half*>(allocate_device_memory(right_bytes));
    try {
        const auto blocks_left = static_cast<unsigned>((left.size() + 255) / 256);
        const auto blocks_right = static_cast<unsigned>((right.size() + 255) / 256);
        float_to_half_kernel<<<blocks_left, 256, 0, compute_stream()>>>(left.device_data(), left_half, left.size());
        float_to_half_kernel<<<blocks_right, 256, 0, compute_stream()>>>(right.device_data(), right_half, right.size());
        checkCuda(cudaGetLastError(), "FP16 conversion kernel launch");

        const float alpha = 1.0f;
        const float beta = 0.0f;
        const auto status = cublasGemmEx(
            cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
            static_cast<int>(right.cols()), static_cast<int>(left.rows()), static_cast<int>(left.cols()),
            &alpha, right_half, CUDA_R_16F, static_cast<int>(right.cols()),
            left_half, CUDA_R_16F, static_cast<int>(left.cols()),
            &beta, output.device_data(), CUDA_R_32F, static_cast<int>(right.cols()),
            CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        if (status != CUBLAS_STATUS_SUCCESS) throw CudaError("FP16 cublasGemmEx failed");
    } catch (...) {
        free_device_memory(left_half);
        free_device_memory(right_half);
        throw;
    }
    free_device_memory(left_half);
    free_device_memory(right_half);
    output.mark_host_stale();
    return output;
}

Matrix matmul_double_accumulate(const Matrix& left, const Matrix& right) {
    if (left.cols() != right.rows()) throw ShapeMismatchError("Matrix dimensions are incompatible for double matmul");
    Matrix output(left.rows(), right.cols());
    if (output.empty()) return output;
    double_accumulate_kernel<<<static_cast<unsigned>((output.size() + 255) / 256), 256, 0, compute_stream()>>>(
        left.device_data(), right.device_data(), output.device_data(), left.rows(), left.cols(), right.cols());
    checkCuda(cudaGetLastError(), "double accumulation kernel launch");
    output.mark_host_stale();
    return output;
}
}
