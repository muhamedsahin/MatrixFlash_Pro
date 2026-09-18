#include "matrix_pro/ops/operations.hpp"
#include "matrix_pro/core/cuda_utils.hpp"
#include "matrix_pro/detail/kernel_helpers.cuh"

#include <cuda_runtime.h>
#include <algorithm>
#include <stdexcept>

namespace matrix_pro {
namespace {

struct DivideOp { __device__ float operator()(float a, float b) const { return a / b; } };
struct DivideScalarOp { float scalar; __device__ float operator()(float a) const { return a / scalar; } };

enum class BroadcastOperation { add, multiply, subtract, divide };

// General 2D broadcasting kernel: each output element maps its row/col to the
// operand index through "match or broadcast from 1" rules.
__global__ void broadcast_kernel(const float* left, const float* right, float* output,
                                 std::size_t rows, std::size_t cols,
                                 std::size_t left_rows, std::size_t left_cols,
                                 std::size_t right_rows, std::size_t right_cols,
                                 BroadcastOperation operation) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto total = rows * cols;
    if (index >= total) return;
    const std::size_t col = index % cols;
    const std::size_t row = index / cols;
    const std::size_t li = (row % left_rows) * left_cols + (col % left_cols);
    const std::size_t ri = (row % right_rows) * right_cols + (col % right_cols);
    if (operation == BroadcastOperation::add) output[index] = left[li] + right[ri];
    else if (operation == BroadcastOperation::multiply) output[index] = left[li] * right[ri];
    else if (operation == BroadcastOperation::subtract) output[index] = left[li] - right[ri];
    else output[index] = left[li] / right[ri];
}

// Sums grad over every dimension in which `target` was broadcast (target dim 1).
__global__ void broadcast_backward_kernel(const float* grad, float* output,
                                          std::size_t rows, std::size_t cols,
                                          std::size_t grad_rows, std::size_t grad_cols,
                                          std::size_t target_rows, std::size_t target_cols) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto total = rows * cols;
    if (index >= total) return;
    const std::size_t col = index % cols;
    const std::size_t row = index / cols;
    float sum = 0.0f;
    if (target_rows == grad_rows) {
        if (target_cols == grad_cols) {
            sum = grad[index];
        } else {
            for (std::size_t gc = 0; gc < grad_cols; ++gc) sum += grad[row * grad_cols + gc];
        }
    } else {
        if (target_cols == grad_cols) {
            for (std::size_t gr = 0; gr < grad_rows; ++gr) sum += grad[gr * grad_cols + col];
        } else {
            for (std::size_t gr = 0; gr < grad_rows; ++gr)
                for (std::size_t gc = 0; gc < grad_cols; ++gc) sum += grad[gr * grad_cols + gc];
        }
    }
    output[index] = sum;
}

__global__ void slice_scatter_kernel(const float* grad, float* output,
                                     std::size_t rows, std::size_t cols,
                                     std::size_t grad_rows, std::size_t grad_cols,
                                     std::size_t row_start, std::size_t col_start) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto total = grad_rows * grad_cols;
    if (index >= total) return;
    const std::size_t col = index % grad_cols;
    const std::size_t row = index / grad_cols;
    output[(row + row_start) * cols + (col + col_start)] = grad[index];
}

__global__ void one_hot_kernel(const float* indices, float* output,
                               std::size_t count, std::size_t classes) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const int label = static_cast<int>(indices[index]);
    if (label >= 0 && static_cast<std::size_t>(label) < classes) output[index * classes + label] = 1.0f;
}

__global__ void copy_strided_kernel(const float* source, float* destination,
                                    std::size_t rows, std::size_t cols,
                                    std::size_t source_cols, std::size_t destination_cols,
                                    std::size_t col_offset) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto total = rows * cols;
    if (index >= total) return;
    const std::size_t col = index % cols;
    const std::size_t row = index / cols;
    destination[row * destination_cols + col_offset + col] = source[row * source_cols + col];
}

void validate_broadcast(const Matrix& left, const Matrix& right) {
    const std::size_t rows = std::max(left.rows(), right.rows());
    const std::size_t cols = std::max(left.cols(), right.cols());
    if ((left.rows() != rows && left.rows() != 1) || (right.rows() != rows && right.rows() != 1) ||
        (left.cols() != cols && left.cols() != 1) || (right.cols() != cols && right.cols() != 1)) {
        throw ShapeMismatchError("Broadcast shapes are incompatible");
    }
}

}

Matrix divide(const Matrix& left, const Matrix& right) {
    if (left.rows() != right.rows() || left.cols() != right.cols())
        throw ShapeMismatchError("Matrix shapes must match");
    Matrix output(left.rows(), left.cols());
    detail::launch_binary(left.device_data(), right.device_data(), output.device_data(),
                          output.size(), DivideOp{}, compute_stream());
    checkCuda(cudaGetLastError(), "divide kernel launch");
    output.mark_host_stale();
    return output;
}

Matrix divide(const Matrix& matrix, float scalar) {
    if (scalar == 0.0f) throw InvalidArgumentError("Division by zero scalar");
    Matrix output(matrix.rows(), matrix.cols());
    detail::launch_unary(matrix.device_data(), output.device_data(), output.size(),
                         DivideScalarOp{scalar}, compute_stream());
    checkCuda(cudaGetLastError(), "divide scalar kernel launch");
    output.mark_host_stale();
    return output;
}

Matrix broadcast_add(const Matrix& left, const Matrix& right) {
    validate_broadcast(left, right);
    Matrix output(std::max(left.rows(), right.rows()), std::max(left.cols(), right.cols()));
    broadcast_kernel<<<static_cast<unsigned>((output.size() + 255) / 256), 256, 0, compute_stream()>>>(
        left.device_data(), right.device_data(), output.device_data(),
        output.rows(), output.cols(), left.rows(), left.cols(), right.rows(), right.cols(),
        BroadcastOperation::add);
    checkCuda(cudaGetLastError(), "broadcast add kernel launch");
    output.mark_host_stale();
    return output;
}

Matrix broadcast_multiply(const Matrix& left, const Matrix& right) {
    validate_broadcast(left, right);
    Matrix output(std::max(left.rows(), right.rows()), std::max(left.cols(), right.cols()));
    broadcast_kernel<<<static_cast<unsigned>((output.size() + 255) / 256), 256, 0, compute_stream()>>>(
        left.device_data(), right.device_data(), output.device_data(),
        output.rows(), output.cols(), left.rows(), left.cols(), right.rows(), right.cols(),
        BroadcastOperation::multiply);
    checkCuda(cudaGetLastError(), "broadcast multiply kernel launch");
    output.mark_host_stale();
    return output;
}

Matrix broadcast_subtract(const Matrix& left, const Matrix& right) {
    validate_broadcast(left, right);
    Matrix output(std::max(left.rows(), right.rows()), std::max(left.cols(), right.cols()));
    broadcast_kernel<<<static_cast<unsigned>((output.size() + 255) / 256), 256, 0, compute_stream()>>>(
        left.device_data(), right.device_data(), output.device_data(),
        output.rows(), output.cols(), left.rows(), left.cols(), right.rows(), right.cols(),
        BroadcastOperation::subtract);
    checkCuda(cudaGetLastError(), "broadcast subtract kernel launch");
    output.mark_host_stale();
    return output;
}

Matrix broadcast_divide(const Matrix& left, const Matrix& right) {
    validate_broadcast(left, right);
    Matrix output(std::max(left.rows(), right.rows()), std::max(left.cols(), right.cols()));
    broadcast_kernel<<<static_cast<unsigned>((output.size() + 255) / 256), 256, 0, compute_stream()>>>(
        left.device_data(), right.device_data(), output.device_data(),
        output.rows(), output.cols(), left.rows(), left.cols(), right.rows(), right.cols(),
        BroadcastOperation::divide);
    checkCuda(cudaGetLastError(), "broadcast divide kernel launch");
    output.mark_host_stale();
    return output;
}

Matrix broadcast_backward(const Matrix& grad, const Matrix& target) {
    validate_broadcast(target, grad);
    Matrix output(target.rows(), target.cols());
    broadcast_backward_kernel<<<static_cast<unsigned>((output.size() + 255) / 256), 256, 0, compute_stream()>>>(
        grad.device_data(), output.device_data(),
        target.rows(), target.cols(), grad.rows(), grad.cols(), target.rows(), target.cols());
    checkCuda(cudaGetLastError(), "broadcast backward kernel launch");
    output.mark_host_stale();
    return output;
}

Matrix reshape(const Matrix& matrix, std::size_t rows, std::size_t cols) {
    if (rows * cols != matrix.size()) throw ShapeMismatchError("Reshape preserves element count");
    Matrix output(rows, cols);
    if (matrix.size() != 0) {
        // Device-to-device copy only. The host mirror is left stale on purpose:
        // copying matrix.data() here would either throw (stale input) or hand
        // the caller a mirror that no longer matches the freshly shaped device
        // buffer. download() materializes it on demand.
        checkCuda(cudaMemcpyAsync(output.device_data(), matrix.device_data(), matrix.size() * sizeof(float),
                                  cudaMemcpyDeviceToDevice, compute_stream()), "reshape device copy");
        output.mark_host_stale();
    }
    return output;
}

Matrix slice_scatter(const Matrix& grad, const Matrix& like,
                     std::size_t row_start, std::size_t row_end,
                     std::size_t col_start, std::size_t col_end) {
    if (row_end < row_start || col_end < col_start) throw OutOfRangeError("Invalid slice bounds");
    if (row_end - row_start != grad.rows() || col_end - col_start != grad.cols())
        throw ShapeMismatchError("Slice scatter gradient shape mismatch");
    if (row_end > like.rows() || col_end > like.cols()) throw OutOfRangeError("Slice scatter out of bounds");
    Matrix output = Matrix::zeros(like.rows(), like.cols());
    if (grad.size() != 0) {
        slice_scatter_kernel<<<static_cast<unsigned>((grad.size() + 255) / 256), 256, 0, compute_stream()>>>(
            grad.device_data(), output.device_data(), like.rows(), like.cols(),
            grad.rows(), grad.cols(), row_start, col_start);
        checkCuda(cudaGetLastError(), "slice scatter kernel launch");
    }
    output.mark_host_stale();
    return output;
}

Matrix one_hot(const Matrix& indices, std::size_t classes) {
    if (classes == 0) throw InvalidArgumentError("one_hot requires at least one class");
    if (indices.rows() != 1 && indices.cols() != 1) throw InvalidArgumentError("one_hot expects a vector of indices");
    Matrix output(indices.size(), classes);
    // Only the hit position is written per row, so clear the buffer first.
    zero_device_memory(output.device_data(), output.size() * sizeof(float));
    one_hot_kernel<<<static_cast<unsigned>((indices.size() + 255) / 256), 256, 0, compute_stream()>>>(
        indices.device_data(), output.device_data(), indices.size(), classes);
    checkCuda(cudaGetLastError(), "one hot kernel launch");
    output.mark_host_stale();
    return output;
}

Matrix concat(const std::vector<Matrix>& parts, std::size_t axis) {
    if (parts.empty()) throw InvalidArgumentError("concat requires at least one matrix");
    if (axis > 1) throw InvalidArgumentError("concat axis must be 0 (rows) or 1 (cols)");
    for (const Matrix& part : parts) {
        if (axis == 0 && part.cols() != parts.front().cols()) throw ShapeMismatchError("concat column mismatch");
        if (axis == 1 && part.rows() != parts.front().rows()) throw ShapeMismatchError("concat row mismatch");
    }
    std::size_t rows = 0, cols = 0;
    for (const Matrix& part : parts) {
        rows += axis == 0 ? part.rows() : 0;
        cols += axis == 1 ? part.cols() : 0;
    }
    rows = axis == 0 ? rows : parts.front().rows();
    cols = axis == 1 ? cols : parts.front().cols();
    Matrix output = Matrix::zeros(rows, cols);
    std::size_t offset = 0;
    for (const Matrix& part : parts) {
        if (part.size() == 0) continue;
        // Both axes copy device-to-device only and leave the host mirror
        // stale; this keeps axis 0 and axis 1 behaviour identical and works
        // with parts whose host mirror is stale (e.g. outputs of GPU ops).
        if (axis == 0) {
            checkCuda(cudaMemcpyAsync(output.device_data() + offset * cols, part.device_data(),
                                      part.size() * sizeof(float), cudaMemcpyDeviceToDevice, compute_stream()),
                      "concat row copy");
            offset += part.rows();
        } else {
            copy_strided_kernel<<<static_cast<unsigned>((part.size() + 255) / 256), 256, 0, compute_stream()>>>(
                part.device_data(), output.device_data(), part.rows(), part.cols(), part.cols(), cols, offset);
            checkCuda(cudaGetLastError(), "concat column copy");
            offset += part.cols();
        }
    }
    output.mark_host_stale();
    return output;
}

Matrix stack(const std::vector<Matrix>& parts) {
    if (parts.empty()) throw InvalidArgumentError("stack requires at least one matrix");
    std::vector<Matrix> flattened;
    flattened.reserve(parts.size());
    for (const Matrix& part : parts) flattened.push_back(reshape(part, 1, part.size()));
    return concat(flattened, 0);
}

Matrix Matrix::operator/(const Matrix& other) const { return divide(*this, other); }
Matrix Matrix::operator/(float scalar) const { return divide(*this, scalar); }
Matrix Matrix::reshape(std::size_t rows, std::size_t cols) const { return matrix_pro::reshape(*this, rows, cols); }

}
