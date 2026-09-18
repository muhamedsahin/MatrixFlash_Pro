#pragma once

// Fused elementwise chains: one kernel + one global-memory round-trip for
// patterns like sigmoid(x)*y, relu(x+y), x*scale+bias, etc.
//
// NOTE: the generic fused_binary/fused_ternary templates are defined in
// src/operations_fused.cu (CUDA-only). To use them with your own functor from
// your own .cu file, include "matrix_pro/detail/fused_ops.cuh" there -- it
// must never be included from a plain .cpp file because it uses CUDA kernel
// launch syntax.

#include <cstddef>

#include "matrix_pro/core/matrix.hpp"

namespace matrix_pro {

// Generic fused binary: out[i] = Op(a[i], b[i]) in a single vectorized pass.
// Op must be trivially copyable with a __device__ operator()(float,float).
// Defined in operations_fused.cu; usable from .cu files including fused.hpp.
template <typename Op>
Matrix fused_binary(const Matrix& a, const Matrix& b, Op op);
template <typename Op>
Matrix& fused_binary_(Matrix& out, const Matrix& a, const Matrix& b, Op op);

// Generic fused ternary: out[i] = Op(a[i], b[i], c[i]).
template <typename Op>
Matrix fused_ternary(const Matrix& a, const Matrix& b, const Matrix& c, Op op);

// --- named hot chains (single kernel each) -----------------------------------
// NOTE: like the fused_binary / fused_ternary templates above, every named chain
// requires its operands to have *identical* shapes -- these kernels are strictly
// elementwise and never broadcast. Use broadcast_add / broadcast_multiply first
// if you need NumPy-style broadcasting.
Matrix fused_sigmoid_mul(const Matrix& x, const Matrix& y);   // sigmoid(x)*y
Matrix fused_relu_add(const Matrix& x, const Matrix& y);      // relu(x+y)
Matrix fused_add_mul(const Matrix& x, const Matrix& y, const Matrix& z); // (x+y)*z
Matrix fused_scale_bias(const Matrix& x, float scale, float bias); // x*s+b
Matrix fused_bias_gelu(const Matrix& x, const Matrix& bias);  // gelu(x+bias)

} // namespace matrix_pro


