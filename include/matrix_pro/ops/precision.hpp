#pragma once

// Mixed-precision GEMM entry points. The storage type of Matrix stays float32;
// these routines convert on the fly (fp16 inputs, fp64 accumulation) so the
// caller keeps the simple float32 API while trading precision for throughput.

#include "matrix_pro/core/matrix.hpp"

namespace matrix_pro {

// FP16 inputs, FP32 output (tensor cores).
Matrix matmul_half(const Matrix& left, const Matrix& right);
// FP64 accumulation, FP32 output (numerically robust reference path).
Matrix matmul_double_accumulate(const Matrix& left, const Matrix& right);

} // namespace matrix_pro
