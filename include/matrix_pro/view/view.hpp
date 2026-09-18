#pragma once

// Non-owning strided view over device memory: zero-copy transpose / slice /
// reshape.
//
// LIFETIME CONTRACT: a view never allocates and never keeps the parent buffer
// alive -- it is a raw pointer plus strides. The parent Matrix must therefore
// outlive every view made from it, and the view is invalidated by anything that
// can move the buffer (destruction, assignment, or a re-allocation through
// upload()/resize-like operations). Materialize into an owning Matrix with
// materialize()/copy_view_to() before the parent goes out of scope.
//
// (The MatrixView constructor accepts an optional owner token so a future
// refactor can make views share ownership of the buffer; no constructor in this
// library currently passes one.)

#include <cstddef>
#include <memory>

#include "matrix_pro/core/errors.hpp"

namespace matrix_pro {

// Forward declaration (full type in matrix.hpp).
class Matrix;

class MatrixView {
public:
    MatrixView() = default;

    // Dense row-major view over an existing device pointer.
    MatrixView(float* data, std::size_t rows, std::size_t cols,
               std::size_t row_stride, std::size_t col_stride,
               std::shared_ptr<void> owner = nullptr) noexcept
        : data_(data), rows_(rows), cols_(cols),
          row_stride_(row_stride), col_stride_(col_stride),
          owner_(std::move(owner)) {}

    float* data() noexcept { return data_; }
    const float* data() const noexcept { return data_; }
    std::size_t rows() const noexcept { return rows_; }
    std::size_t cols() const noexcept { return cols_; }
    std::size_t row_stride() const noexcept { return row_stride_; }
    std::size_t col_stride() const noexcept { return col_stride_; }
    std::size_t size() const noexcept { return rows_ * cols_; }
    bool empty() const noexcept { return size() == 0; }
    // True when the view is dense row-major and can use vectorized kernels.
    bool contiguous() const noexcept {
        return col_stride_ == 1 && (rows_ <= 1 || row_stride_ == cols_);
    }

    // Element address for a (row, col) pair. Host-side helper for kernels.
    float* at_ptr(std::size_t row, std::size_t col) noexcept {
        return data_ + row * row_stride_ + col * col_stride_;
    }
    const float* at_ptr(std::size_t row, std::size_t col) const noexcept {
        return data_ + row * row_stride_ + col * col_stride_;
    }

private:
    float* data_ = nullptr;
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::size_t row_stride_ = 0;
    std::size_t col_stride_ = 0;
    std::shared_ptr<void> owner_;
};

// --- view constructors (all zero-copy) -------------------------------------
// Transposed view: (rows x cols) -> (cols x rows) by swapping strides.
MatrixView transpose_view(Matrix& matrix) noexcept;
// Sub-matrix view [row_start,row_end) x [col_start,col_end).
MatrixView slice_view(Matrix& matrix, std::size_t row_start, std::size_t row_end,
                      std::size_t col_start, std::size_t col_end);
// Reshape view; only valid for contiguous matrices (throws otherwise).
MatrixView reshape_view(Matrix& matrix, std::size_t rows, std::size_t cols);
// Fully general strided view (offset + custom strides, bounds-checked).
MatrixView as_strided_view(Matrix& matrix, std::size_t rows, std::size_t cols,
                           std::size_t offset, std::size_t row_stride,
                           std::size_t col_stride);

// --- view utilities ----------------------------------------------------------
// Materialize a (possibly strided) view into a new dense Matrix (one copy).
Matrix materialize(const MatrixView& view);
// Copy view contents into an existing dense matrix of matching shape.
void copy_view_to(const MatrixView& src, Matrix& dst);
// Elementwise add/mul of two views into a dense output (broadcast-aware).
Matrix add_views(const MatrixView& a, const MatrixView& b);
Matrix multiply_views(const MatrixView& a, const MatrixView& b);

} // namespace matrix_pro
