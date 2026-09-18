#include "matrix_pro/indexing/indexing.hpp"
#include "matrix_pro/core/matrix.hpp"
#include "matrix_pro/core/cuda_utils.hpp"

#include <cuda_runtime.h>
#include <cmath>

namespace matrix_pro {
namespace {

__global__ void index_select_axis0_kernel(const float* src, const float* idx, float* dst,
                                          std::size_t cols, std::size_t k) {
    const std::size_t t = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= k * cols) return;
    const std::size_t i = t / cols;
    const std::size_t c = t % cols;
    const std::size_t row = static_cast<std::size_t>(idx[i]);
    dst[t] = src[row * cols + c];
}

__global__ void index_select_axis1_kernel(const float* src, const float* idx, float* dst,
                                          std::size_t rows, std::size_t cols, std::size_t k) {
    const std::size_t t = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= rows * k) return;
    const std::size_t r = t / k;
    const std::size_t j = t % k;
    const std::size_t col = static_cast<std::size_t>(idx[j]);
    dst[t] = src[r * cols + col];
}

__global__ void scatter_add_kernel(const float* src, const float* idx, float* dst,
                                   std::size_t cols, std::size_t n) {
    const std::size_t t = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= n * cols) return;
    const std::size_t i = t / cols;
    const std::size_t c = t % cols;
    const std::size_t row = static_cast<std::size_t>(idx[i]);
    atomicAdd(&dst[row * cols + c], src[t]);
}

} // namespace

Matrix index_select(const Matrix& matrix, const Matrix& indices, std::size_t axis) {
    if (axis > 1) throw InvalidArgumentError("index_select axis must be 0 or 1");
    if (indices.rows() != 1 && indices.cols() != 1)
        throw InvalidArgumentError("index_select expects a vector of indices");
    const std::size_t bound = (axis == 0) ? matrix.rows() : matrix.cols();
    const std::size_t k = indices.size();
    Matrix copy = indices;
    copy.download();
    for (float v : copy.data()) {
        if (!std::isfinite(v) || v < 0 || static_cast<std::size_t>(v) >= bound)
            throw OutOfRangeError("index_select index out of range");
    }
    Matrix out = (axis == 0) ? Matrix(k, matrix.cols()) : Matrix(matrix.rows(), k);
    if (out.empty()) return out;
    if (axis == 0) {
        index_select_axis0_kernel<<<(out.size() + 255) / 256, 256, 0, compute_stream()>>>(
            matrix.device_data(), indices.device_data(), out.device_data(),
            matrix.cols(), k);
    } else {
        index_select_axis1_kernel<<<(out.size() + 255) / 256, 256, 0, compute_stream()>>>(
            matrix.device_data(), indices.device_data(), out.device_data(),
            matrix.rows(), matrix.cols(), k);
    }
    checkCuda(cudaGetLastError(), "index_select kernel launch");
    out.mark_host_stale();
    return out;
}

Matrix gather(const Matrix& matrix, const Matrix& indices, std::size_t axis) {
    return index_select(matrix, indices, axis);
}

Matrix scatter_add(const Matrix& src, const Matrix& index, std::size_t rows) {
    if (index.rows() != 1 && index.cols() != 1)
        throw InvalidArgumentError("scatter_add index must be a vector");
    if (src.rows() != index.size())
        throw ShapeMismatchError("scatter_add src rows must match index count");
    Matrix out = Matrix::zeros(rows, src.cols());
    if (src.empty()) return out;
    Matrix idx_copy = index;
    idx_copy.download();
    for (float v : idx_copy.data()) {
        if (!std::isfinite(v) || static_cast<std::size_t>(v) >= rows)
            throw OutOfRangeError("scatter_add index out of range");
    }
    scatter_add_kernel<<<(src.size() + 255) / 256, 256, 0, compute_stream()>>>(
        src.device_data(), index.device_data(), out.device_data(),
        src.cols(), index.size());
    checkCuda(cudaGetLastError(), "scatter_add kernel launch");
    out.mark_host_stale();
    return out;
}

Matrix embedding(const Matrix& weight, const Matrix& indices) {
    return index_select(weight, indices, 0);
}

Matrix embedding_backward(const Matrix& grad, const Matrix& indices,
                          std::size_t vocab_size) {
    return scatter_add(grad, indices, vocab_size);
}

Matrix scatter(const Matrix& src, const Matrix& indices, std::size_t rows,
               std::size_t cols) {
    if (indices.rows() != 1 && indices.cols() != 1)
        throw InvalidArgumentError("scatter indices must be a vector");
    if (src.size() != indices.size())
        throw ShapeMismatchError("scatter src and indices must match in count");
    Matrix out = Matrix::zeros(rows, cols);
    if (src.empty()) return out;
    Matrix idx_copy = indices;
    idx_copy.download();
    for (float v : idx_copy.data()) {
        if (!std::isfinite(v) || static_cast<std::size_t>(v) >= rows * cols)
            throw OutOfRangeError("scatter index out of range");
    }
    scatter_add_kernel<<<(src.size() + 255) / 256, 256, 0, compute_stream()>>>(
        src.device_data(), indices.device_data(), out.device_data(), 1, indices.size());
    checkCuda(cudaGetLastError(), "scatter kernel launch");
    out.mark_host_stale();
    return out;
}

} // namespace matrix_pro
