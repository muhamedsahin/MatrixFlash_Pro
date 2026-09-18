#pragma once

// Gather / scatter / embedding primitives for attention & embedding layers.
// All indices are float-encoded integers in (n x 1) or (1 x n) matrices,
// matching the existing one_hot() convention.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "matrix_pro/core/matrix.hpp"

namespace matrix_pro {


// Rows/cols picked by `indices` along `axis` (0 = rows, 1 = cols).
// indices: (k x 1) or (1 x k); out-of-range index throws OutOfRangeError.
Matrix index_select(const Matrix& matrix, const Matrix& indices, std::size_t axis);
// Alias with PyTorch-style naming.
Matrix gather(const Matrix& matrix, const Matrix& indices, std::size_t axis);
// Scatter src rows into `rows` output rows with atomicAdd accumulation:
//     out[index[i]][c] += src[i][c]
// i.e. index[i] is the DESTINATION row of source row i, so duplicate indices
// accumulate (this is exactly what embedding_backward needs).
// Shapes: src (k x cols), index (k x 1) or (1 x k) -> out (rows x cols).
Matrix scatter_add(const Matrix& src, const Matrix& index, std::size_t rows);
// Embedding lookup: weight (vocab x dim), indices (n x 1)->(n x dim).
Matrix embedding(const Matrix& weight, const Matrix& indices);
// Gradient of embedding: accumulates grad rows back into a vocab x dim zero
// matrix (scatter-add of grad by indices).
Matrix embedding_backward(const Matrix& grad, const Matrix& indices,
                          std::size_t vocab_size);
// Scatter a patch back into a zero matrix shaped like `like`
// (device-side counterpart of slice_scatter for strided updates).
Matrix scatter(const Matrix& src, const Matrix& indices, std::size_t rows,
               std::size_t cols);

} // namespace matrix_pro
