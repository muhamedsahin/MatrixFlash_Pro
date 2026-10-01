// Dahili uygulama parçası: src/ops/shape.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Broadcast mapping, scatter-gradient, one-hot ve strided copy.

namespace {

struct DivideOp {
    __device__ float operator()(float a, float b) const { return a / b; }
};
struct DivideScalarOp {
    float scalar;
    __device__ float operator()(float a) const { return a / scalar; }
};

enum class BroadcastOperation { add, multiply, subtract, divide };

// General 2D broadcasting kernel: each output element maps its row/col to the
// operand index through "match or broadcast from 1" rules.
__global__ void broadcast_kernel(const float* left, const float* right, float* output,
                                 std::size_t rows, std::size_t cols, std::size_t left_rows,
                                 std::size_t left_cols, std::size_t right_rows,
                                 std::size_t right_cols, BroadcastOperation operation) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto total = rows * cols;
    if (index >= total)
        return;
    const std::size_t col = index % cols;
    const std::size_t row = index / cols;
    const std::size_t li = (row % left_rows) * left_cols + (col % left_cols);
    const std::size_t ri = (row % right_rows) * right_cols + (col % right_cols);
    if (operation == BroadcastOperation::add)
        output[index] = left[li] + right[ri];
    else if (operation == BroadcastOperation::multiply)
        output[index] = left[li] * right[ri];
    else if (operation == BroadcastOperation::subtract)
        output[index] = left[li] - right[ri];
    else
        output[index] = left[li] / right[ri];
}

// Sums grad over every dimension in which `target` was broadcast (target dim 1).
__global__ void broadcast_backward_kernel(const float* grad, float* output, std::size_t rows,
                                          std::size_t cols, std::size_t grad_rows,
                                          std::size_t grad_cols, std::size_t target_rows,
                                          std::size_t target_cols) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto total = rows * cols;
    if (index >= total)
        return;
    const std::size_t col = index % cols;
    const std::size_t row = index / cols;
    float sum = 0.0f;
    if (target_rows == grad_rows) {
        if (target_cols == grad_cols) {
            sum = grad[index];
        } else {
            for (std::size_t gc = 0; gc < grad_cols; ++gc)
                sum += grad[row * grad_cols + gc];
        }
    } else {
        if (target_cols == grad_cols) {
            for (std::size_t gr = 0; gr < grad_rows; ++gr)
                sum += grad[gr * grad_cols + col];
        } else {
            for (std::size_t gr = 0; gr < grad_rows; ++gr)
                for (std::size_t gc = 0; gc < grad_cols; ++gc)
                    sum += grad[gr * grad_cols + gc];
        }
    }
    output[index] = sum;
}

__global__ void slice_scatter_kernel(const float* grad, float* output, std::size_t rows,
                                     std::size_t cols, std::size_t grad_rows, std::size_t grad_cols,
                                     std::size_t row_start, std::size_t col_start) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto total = grad_rows * grad_cols;
    if (index >= total)
        return;
    const std::size_t col = index % grad_cols;
    const std::size_t row = index / grad_cols;
    output[(row + row_start) * cols + (col + col_start)] = grad[index];
}

__global__ void one_hot_kernel(const float* indices, float* output, std::size_t count,
                               std::size_t classes) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count)
        return;
    const int label = static_cast<int>(indices[index]);
    if (label >= 0 && static_cast<std::size_t>(label) < classes)
        output[index * classes + label] = 1.0f;
}

__global__ void copy_strided_kernel(const float* source, float* destination, std::size_t rows,
                                    std::size_t cols, std::size_t source_cols,
                                    std::size_t destination_cols, std::size_t col_offset) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto total = rows * cols;
    if (index >= total)
        return;
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
