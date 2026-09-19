#include "matrix_pro/detail/gemm/gemm_api.hpp"
#include "matrix_pro/core/cuda_utils.hpp"

#include <cublas_v2.h>

namespace matrix_pro {
namespace detail {
namespace gemm {

void gemm_cublas_ex(const float* a, const float* b, float* c,
                    int m, int n, int k, cudaStream_t stream) {
    // Row-major (m×k)*(k×n) == column-major (n×m) with operands swapped.
    // Stream is already bound on the per-thread handle at context creation —
    // only re-bind if the caller passes a different stream (rare).
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasHandle_t handle = cublas_handle();
    if (stream != compute_stream()) {
        cublasSetStream(handle, stream);
    }
    // TENSOR_OP forces the TF32 / tensor-core path on Ampere+ instead of
    // letting the legacy SIMT heuristic win on medium squares.
    const cublasStatus_t status = cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        n, m, k,
        &alpha,
        b, CUDA_R_32F, n,
        a, CUDA_R_32F, k,
        &beta,
        c, CUDA_R_32F, n,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (status != CUBLAS_STATUS_SUCCESS) {
        // Older drivers / GPUs without tensor-op may reject TENSOR_OP; fall back.
        const cublasStatus_t retry = cublasGemmEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            n, m, k,
            &alpha,
            b, CUDA_R_32F, n,
            a, CUDA_R_32F, k,
            &beta,
            c, CUDA_R_32F, n,
            CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
        if (retry != CUBLAS_STATUS_SUCCESS) {
            throw CudaError("cublasGemmEx failed");
        }
    }
}

} // namespace gemm
} // namespace detail
} // namespace matrix_pro
