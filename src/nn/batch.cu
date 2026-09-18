#include "matrix_pro/ops/operations.hpp"
#include "matrix_pro/core/tensor.hpp"
#include "matrix_pro/core/cuda_utils.hpp"

#include <cublas_v2.h>
#include <stdexcept>

namespace matrix_pro {
namespace {
}

Tensor batch_matmul(const Tensor& left, const Tensor& right) {
    if (left.rank() != 3 || right.rank() != 3) throw InvalidArgumentError("batch_matmul requires rank-3 tensors");
    const auto& a = left.shape();
    const auto& b = right.shape();
    if (a[0] != b[0] || a[2] != b[1]) throw ShapeMismatchError("batch_matmul shape mismatch");
    Tensor output({a[0], a[1], b[2]});
    if (output.empty()) return output;

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const long long stride_a = static_cast<long long>(a[1] * a[2]);
    const long long stride_b = static_cast<long long>(b[1] * b[2]);
    const long long stride_c = static_cast<long long>(a[1] * b[2]);
    const auto status = cublasSgemmStridedBatched(
        cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
        static_cast<int>(b[2]), static_cast<int>(a[1]), static_cast<int>(a[2]),
        &alpha, right.device_data(), static_cast<int>(b[2]), stride_b,
        left.device_data(), static_cast<int>(a[2]), stride_a,
        &beta, output.device_data(), static_cast<int>(b[2]), stride_c,
        static_cast<int>(a[0]));
    if (status != CUBLAS_STATUS_SUCCESS) throw CudaError("cublasSgemmStridedBatched failed");
    output.mark_host_stale();
    return output;
}
}