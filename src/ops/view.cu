#include "matrix_pro/view/view.hpp"
#include "matrix_pro/core/matrix.hpp"
#include "matrix_pro/core/cuda_utils.hpp"
#include "matrix_pro/detail/kernel_helpers.cuh"

#include <cuda_runtime.h>

namespace matrix_pro {
namespace {

__global__ void materialize_kernel(const float* src, float* dst,
                                   std::size_t rows, std::size_t cols,
                                   std::size_t row_stride, std::size_t col_stride) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= rows * cols) return;
    const std::size_t r = index / cols;
    const std::size_t c = index % cols;
    dst[index] = src[r * row_stride + c * col_stride];
}

__global__ void view_binary_kernel(const float* a, float* out,
                                   std::size_t rows, std::size_t cols,
                                   std::size_t a_rs, std::size_t a_cs,
                                   const float* b, std::size_t b_rs, std::size_t b_cs,
                                   int is_add) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= rows * cols) return;
    const std::size_t r = index / cols;
    const std::size_t c = index % cols;
    const float x = a[r * a_rs + c * a_cs];
    const float y = b[r * b_rs + c * b_cs];
    out[index] = is_add ? (x + y) : (x * y);
}

} // namespace

MatrixView transpose_view(Matrix& matrix) noexcept {
    if (matrix.empty()) return MatrixView(nullptr, 0, 0, 0, 0);
    // (r,c) element lives at data[r*C+c]; transposed (c,r) uses strides (1,C).
    return MatrixView(matrix.device_data(), matrix.cols(), matrix.rows(),
                      1, matrix.cols());
}

MatrixView slice_view(Matrix& matrix, std::size_t row_start, std::size_t row_end,
                      std::size_t col_start, std::size_t col_end) {
    if (row_start > row_end || col_start > col_end ||
        row_end > matrix.rows() || col_end > matrix.cols())
        throw OutOfRangeError("slice_view range out of bounds");
    float* base = matrix.device_data() + row_start * matrix.cols() + col_start;
    return MatrixView(base, row_end - row_start, col_end - col_start,
                      matrix.cols(), 1);
}

MatrixView reshape_view(Matrix& matrix, std::size_t rows, std::size_t cols) {
    if (rows * cols != matrix.size())
        throw ShapeMismatchError("reshape_view element count mismatch");
    // Only dense row-major buffers can be reshaped without a copy.
    return MatrixView(matrix.device_data(), rows, cols, cols, 1);
}

MatrixView as_strided_view(Matrix& matrix, std::size_t rows, std::size_t cols,
                           std::size_t offset, std::size_t row_stride,
                           std::size_t col_stride) {
    if (rows == 0 || cols == 0) return MatrixView(matrix.device_data(), 0, 0, row_stride, col_stride);
    const std::size_t last = offset + (rows - 1) * row_stride + (cols - 1) * col_stride;
    if (last >= matrix.size())
        throw OutOfRangeError("as_strided_view escapes parent storage");
    return MatrixView(matrix.device_data() + offset, rows, cols, row_stride, col_stride);
}

Matrix materialize(const MatrixView& view) {
    Matrix out(view.rows(), view.cols());
    if (view.empty()) return out;
    if (view.contiguous()) {
        checkCuda(cudaMemcpyAsync(out.device_data(), view.data(),
                                  view.size() * sizeof(float),
                                  cudaMemcpyDeviceToDevice, compute_stream()),
                  "materialize contiguous copy");
        out.mark_host_stale();
        return out;
    }
    materialize_kernel<<<detail::grid_blocks(view.size()), 256, 0, compute_stream()>>>(
        view.data(), out.device_data(), view.rows(), view.cols(),
        view.row_stride(), view.col_stride());
    checkCuda(cudaGetLastError(), "materialize kernel launch");
    out.mark_host_stale();
    return out;
}

void copy_view_to(const MatrixView& src, Matrix& dst) {
    if (src.rows() != dst.rows() || src.cols() != dst.cols())
        throw ShapeMismatchError("copy_view_to shape mismatch");
    if (src.empty()) return;
    if (src.contiguous()) {
        checkCuda(cudaMemcpyAsync(dst.device_data(), src.data(),
                                  src.size() * sizeof(float),
                                  cudaMemcpyDeviceToDevice, compute_stream()),
                  "copy_view_to contiguous copy");
    } else {
        materialize_kernel<<<detail::grid_blocks(src.size()), 256, 0, compute_stream()>>>(
            src.data(), dst.device_data(), src.rows(), src.cols(),
            src.row_stride(), src.col_stride());
        checkCuda(cudaGetLastError(), "copy_view_to kernel launch");
    }
    dst.mark_host_stale();  // the device buffer now holds newer data than the mirror
}

Matrix add_views(const MatrixView& a, const MatrixView& b) {
    if (a.rows() != b.rows() || a.cols() != b.cols())
        throw ShapeMismatchError("add_views shape mismatch");
    Matrix out(a.rows(), a.cols());
    if (a.empty()) return out;
    view_binary_kernel<<<detail::grid_blocks(a.size()), 256, 0, compute_stream()>>>(
        a.data(), out.device_data(), a.rows(), a.cols(),
        a.row_stride(), a.col_stride(), b.data(), b.row_stride(), b.col_stride(), 1);
    checkCuda(cudaGetLastError(), "add_views kernel launch");
    out.mark_host_stale();
    return out;
}

Matrix multiply_views(const MatrixView& a, const MatrixView& b) {
    if (a.rows() != b.rows() || a.cols() != b.cols())
        throw ShapeMismatchError("multiply_views shape mismatch");
    Matrix out(a.rows(), a.cols());
    if (a.empty()) return out;
    view_binary_kernel<<<detail::grid_blocks(a.size()), 256, 0, compute_stream()>>>(
        a.data(), out.device_data(), a.rows(), a.cols(),
        a.row_stride(), a.col_stride(), b.data(), b.row_stride(), b.col_stride(), 0);
    checkCuda(cudaGetLastError(), "multiply_views kernel launch");
    out.mark_host_stale();
    return out;
}

} // namespace matrix_pro
