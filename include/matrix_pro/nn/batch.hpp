#pragma once

// Batched GEMM for rank-3 tensors plus the Tensor elementwise helpers the
// autograd engine builds on.

#include "matrix_pro/core/tensor.hpp"

namespace matrix_pro {

// Batched GEMM: a=[batch,m,k], b=[batch,k,n], result=[batch,m,n].
Tensor batch_matmul(const Tensor& left, const Tensor& right);
// Backward of batch_matmul: dL/dA and dL/dB given the upstream gradient.
Tensor batch_matmul_backward_left(const Tensor& grad, const Tensor& right);   // G x B^T
Tensor batch_matmul_backward_right(const Tensor& grad, const Tensor& left);   // A^T x G

// --- tensor elementwise (autograd support) ---
Tensor tensor_add(const Tensor& left, const Tensor& right);
Tensor tensor_multiply(const Tensor& left, const Tensor& right);
Tensor tensor_multiply(const Tensor& tensor, float scalar);
Tensor tensor_negate(const Tensor& tensor);

} // namespace matrix_pro