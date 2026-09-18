#include "matrix_pro/ops/operations.hpp"
#include "matrix_pro/core/cuda_utils.hpp"

#include <cublas_v2.h>
#include <stdexcept>

namespace matrix_pro {
namespace {

__global__ void outer_product_kernel(const float* left, const float* right, float* output,
                                     std::size_t left_count, std::size_t right_count) {
    const auto col = blockIdx.x * blockDim.x + threadIdx.x;
    const auto row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < left_count && col < right_count) output[row * right_count + col] = left[row] * right[col];
}

}

Matrix multiply(const Matrix& left, const Matrix& right) {
    if (left.cols() != right.rows()) {
        throw ShapeMismatchError("Matrix dimensions are incompatible for multiplication");
    }
    if (left.rows() * right.cols() == 0) return Matrix(left.rows(), right.cols());

    const int m = left.rows();
    const int n = right.cols();
    const int k = left.cols();
    const float alpha = 1.0f;
    const float beta = 0.0f;

    Matrix output(left.rows(), right.cols());
    // cublasGemmEx lets cuBLAS pick the best kernel (tensor-op / split-k /
    // atomics) per shape through its internal heuristics instead of locking in
    // the legacy sgemm path. Combined with CUBLAS_TF32_TENSOR_OP_MATH on
    // Ampere+ (set in the execution context) this is the fastest fp32 path.
    const cublasStatus_t status = cublasGemmEx(
        cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
        n, m, k,
        &alpha,
        right.device_data(), CUDA_R_32F, n,
        left.device_data(), CUDA_R_32F, k,
        &beta,
        output.device_data(), CUDA_R_32F, n,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw CudaError("cuBLAS gemm failed");
    }
    output.mark_host_stale();
    return output;
}

Matrix outer_product(const Matrix& left, const Matrix& right) {
    if (left.size() == 0 || right.size() == 0) return Matrix(left.size(), right.size());
    const std::size_t lc = left.size();
    const std::size_t rc = right.size();
    Matrix output(lc, rc);
    dim3 block(16, 16);
    dim3 grid((rc + 15) / 16, (lc + 15) / 16);
    outer_product_kernel<<<grid, block, 0, compute_stream()>>>(left.device_data(), right.device_data(), output.device_data(), lc, rc);
    checkCuda(cudaGetLastError(), "outer product kernel launch");
    output.mark_host_stale();
    return output;
}

Matrix Matrix::operator*(const Matrix& other) const { return matrix_pro::multiply(*this, other); }
Matrix Matrix::outer_product(const Matrix& other) const { return matrix_pro::outer_product(*this, other); }

}