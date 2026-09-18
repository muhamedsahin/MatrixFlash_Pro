#pragma once

// Matrix products: cuBLAS GEMM plus the outer / Kronecker variants.
// Mixed-precision GEMM variants live in matrix_pro/ops/precision.hpp and the
// batched (Tensor) GEMM in matrix_pro/nn/batch.hpp.

#include <cstddef>

#include "matrix_pro/core/matrix.hpp"

namespace matrix_pro {

// cuBLAS-backed matrix multiplication (uses the tensor-core fast path when the
// problem is large enough).
Matrix multiply(const Matrix& left, const Matrix& right);

// Outer product of two vectors: (m x 1) * (1 x n) -> (m x n).
Matrix outer_product(const Matrix& left, const Matrix& right);
// Kronecker product: (a x b) kron (c x d) -> (a*c x b*d).
Matrix kron(const Matrix& left, const Matrix& right);

} // namespace matrix_pro
