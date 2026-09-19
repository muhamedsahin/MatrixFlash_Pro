#pragma once

// Internal row-major GEMM entry points. Public Matrix::operator* routes here.
// Layout: C[m,n] = A[m,k] * B[k,n]  (row-major, contiguous).

#include <cstddef>

#include <cuda_runtime.h>

namespace matrix_pro {
namespace detail {
namespace gemm {

enum class Epilogue {
    none = 0,
    bias,       // C = AB + bias[col]
    bias_relu,  // C = relu(AB + bias[col])
    bias_gelu   // C = gelu(AB + bias[col])
};

// Shape-aware dispatcher: picks micro / gemv / tiled / cublas / cublasLt.
void gemm_rowmajor(const float* a, const float* b, float* c,
                   int m, int n, int k, cudaStream_t stream);

// In-place epilogue after a plain GEMM (used when fused Lt path unavailable).
void apply_epilogue(float* c, const float* bias, int m, int n,
                    Epilogue epi, cudaStream_t stream);

// Fused GEMM + column bias + optional activation (cublasLt epilogue when possible,
// otherwise GEMM + vectorized epilogue kernel).
void gemm_bias_epilogue(const float* a, const float* b, float* c,
                        const float* bias, int m, int n, int k,
                        Epilogue epi, cudaStream_t stream);

// Backend implementations (called by the dispatcher).
void gemm_cublas_ex(const float* a, const float* b, float* c,
                    int m, int n, int k, cudaStream_t stream);
void gemm_cublas_lt(const float* a, const float* b, float* c,
                    int m, int n, int k, cudaStream_t stream);
void gemm_micro(const float* a, const float* b, float* c,
                int m, int n, int k, cudaStream_t stream);
void gemm_gemv(const float* a, const float* b, float* c,
               int m, int n, int k, cudaStream_t stream);
void gemm_tiled(const float* a, const float* b, float* c,
                int m, int n, int k, cudaStream_t stream);

} // namespace gemm
} // namespace detail
} // namespace matrix_pro
