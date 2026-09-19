#pragma once

// Matrix products: cuBLAS GEMM plus the outer / Kronecker variants.
// Mixed-precision GEMM variants live in matrix_pro/ops/precision.hpp and the
// batched (Tensor) GEMM in matrix_pro/nn/batch.hpp.

#include <cstddef>

#include "matrix_pro/core/matrix.hpp"

namespace matrix_pro {

// Shape-aware GEMM: micro / GEMV / cublasLt / cuBLAS. Output is device_only.
Matrix multiply(const Matrix& left, const Matrix& right);

// Zero-allocation hot path: writes into a pre-sized output (must be m×n).
void multiply_into(const Matrix& left, const Matrix& right, Matrix& output);

// Fused GEMM + column bias + activation (cublasLt epilogue when available).
// Beats separate multiply() + elementwise on the critical MLP path.
Matrix gemm_bias_relu(const Matrix& left, const Matrix& right, const Matrix& bias);
Matrix gemm_bias_gelu(const Matrix& left, const Matrix& right, const Matrix& bias);

// Outer product of two vectors: (m x 1) * (1 x n) -> (m x n).
Matrix outer_product(const Matrix& left, const Matrix& right);
// Kronecker product: (a x b) kron (c x d) -> (a*c x b*d).
Matrix kron(const Matrix& left, const Matrix& right);

} // namespace matrix_pro
