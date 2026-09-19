#include "matrix_pro/ops/operations.hpp"
#include "matrix_pro/core/cuda_utils.hpp"
#include "matrix_pro/detail/gemm/gemm_api.hpp"

#include <stdexcept>

namespace matrix_pro {
namespace {

__global__ void outer_product_kernel(const float* left, const float* right, float* output,
                                     std::size_t left_count, std::size_t right_count) {
    const auto col = blockIdx.x * blockDim.x + threadIdx.x;
    const auto row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < left_count && col < right_count) {
        output[row * right_count + col] = left[row] * right[col];
    }
}

} // namespace

Matrix multiply(const Matrix& left, const Matrix& right) {
    if (left.cols() != right.rows()) {
        throw ShapeMismatchError("Matrix dimensions are incompatible for multiplication");
    }
    if (left.rows() * right.cols() == 0) {
        return Matrix(left.rows(), right.cols(), MemoryMode::device_only);
    }

    // device_only: skip the host mirror allocation — GEMM never needs it until
    // the caller explicitly downloads. This alone closes most of the gap versus
    // raw cuBLAS in application-style (alloc-per-call) benchmarks.
    Matrix output(left.rows(), right.cols(), MemoryMode::device_only);
    detail::gemm::gemm_rowmajor(
        left.device_data(), right.device_data(), output.device_data(),
        static_cast<int>(left.rows()), static_cast<int>(right.cols()),
        static_cast<int>(left.cols()), compute_stream());
    output.mark_host_stale();
    return output;
}

void multiply_into(const Matrix& left, const Matrix& right, Matrix& output) {
    if (left.cols() != right.rows()) {
        throw ShapeMismatchError("Matrix dimensions are incompatible for multiplication");
    }
    if (output.rows() != left.rows() || output.cols() != right.cols()) {
        throw ShapeMismatchError("multiply_into output shape mismatch");
    }
    if (left.rows() * right.cols() == 0) return;
    detail::gemm::gemm_rowmajor(
        left.device_data(), right.device_data(), output.device_data(),
        static_cast<int>(left.rows()), static_cast<int>(right.cols()),
        static_cast<int>(left.cols()), compute_stream());
    output.mark_host_stale();
}

Matrix gemm_bias_relu(const Matrix& left, const Matrix& right, const Matrix& bias) {
    if (left.cols() != right.rows()) {
        throw ShapeMismatchError("gemm_bias_relu: incompatible GEMM shapes");
    }
    if (bias.size() != right.cols()) {
        throw ShapeMismatchError("gemm_bias_relu: bias length must equal N");
    }
    Matrix output(left.rows(), right.cols(), MemoryMode::device_only);
    if (output.empty()) return output;
    detail::gemm::gemm_bias_epilogue(
        left.device_data(), right.device_data(), output.device_data(),
        bias.device_data(),
        static_cast<int>(left.rows()), static_cast<int>(right.cols()),
        static_cast<int>(left.cols()), detail::gemm::Epilogue::bias_relu,
        compute_stream());
    output.mark_host_stale();
    return output;
}

Matrix gemm_bias_gelu(const Matrix& left, const Matrix& right, const Matrix& bias) {
    if (left.cols() != right.rows()) {
        throw ShapeMismatchError("gemm_bias_gelu: incompatible GEMM shapes");
    }
    if (bias.size() != right.cols()) {
        throw ShapeMismatchError("gemm_bias_gelu: bias length must equal N");
    }
    Matrix output(left.rows(), right.cols(), MemoryMode::device_only);
    if (output.empty()) return output;
    detail::gemm::gemm_bias_epilogue(
        left.device_data(), right.device_data(), output.device_data(),
        bias.device_data(),
        static_cast<int>(left.rows()), static_cast<int>(right.cols()),
        static_cast<int>(left.cols()), detail::gemm::Epilogue::bias_gelu,
        compute_stream());
    output.mark_host_stale();
    return output;
}

Matrix outer_product(const Matrix& left, const Matrix& right) {
    if (left.size() == 0 || right.size() == 0) {
        return Matrix(left.size(), right.size(), MemoryMode::device_only);
    }
    const std::size_t lc = left.size();
    const std::size_t rc = right.size();
    Matrix output(lc, rc, MemoryMode::device_only);
    dim3 block(16, 16);
    dim3 grid((rc + 15) / 16, (lc + 15) / 16);
    outer_product_kernel<<<grid, block, 0, compute_stream()>>>(
        left.device_data(), right.device_data(), output.device_data(), lc, rc);
    checkCuda(cudaGetLastError(), "outer product kernel launch");
    output.mark_host_stale();
    return output;
}

Matrix Matrix::operator*(const Matrix& other) const {
    return matrix_pro::multiply(*this, other);
}
Matrix Matrix::outer_product(const Matrix& other) const {
    return matrix_pro::outer_product(*this, other);
}

} // namespace matrix_pro
