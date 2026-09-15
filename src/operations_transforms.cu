#include "matrix_pro/operations.hpp"
#include "matrix_pro/cuda_utils.hpp"

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace matrix_pro {
namespace {

constexpr unsigned tile_size = 16;
constexpr unsigned softmax_block = 256;

__global__ void transpose_kernel(const float* input, float* output, std::size_t rows, std::size_t cols) {
    const auto col = blockIdx.x * blockDim.x + threadIdx.x;
    const auto row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && col < cols) output[col * rows + row] = input[row * cols + col];
}

__global__ void relu_kernel(const float* input, float* output, std::size_t count) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) output[index] = input[index] > 0.0f ? input[index] : 0.0f;
}

__global__ void softmax_row_max_kernel(const float* input, std::size_t rows, std::size_t cols, float* row_max) {
    __shared__ float shared[softmax_block];
    const auto tid = threadIdx.x;
    const auto row = blockIdx.x;
    const auto base = row * cols;
    float local = -FLT_MAX;

    for (std::size_t col = tid; col < cols; col += blockDim.x) {
        local = fmaxf(local, input[base + col]);
    }
    shared[tid] = local;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        __syncthreads();
    }
    if (tid == 0) row_max[row] = shared[0];
}

__global__ void softmax_row_sum_kernel(const float* input, float* output, std::size_t rows,
                                      std::size_t cols, const float* row_max, float* row_total) {
    __shared__ float shared[softmax_block];
    const auto tid = threadIdx.x;
    const auto row = blockIdx.x;
    const auto base = row * cols;
    float local = 0.0f;
    const float maximum = row_max[row];

    for (std::size_t col = tid; col < cols; col += blockDim.x) {
        const float value = expf(input[base + col] - maximum);
        output[base + col] = value;
        local += value;
    }
    shared[tid] = local;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared[tid] += shared[tid + stride];
        __syncthreads();
    }
    if (tid == 0) row_total[row] = shared[0];
}

__global__ void softmax_row_norm_kernel(float* output, std::size_t rows, std::size_t cols, const float* row_total) {
    const auto row = blockIdx.x;
    const auto base = row * cols;
    const float total = row_total[row];
    for (std::size_t col = threadIdx.x; col < cols; col += blockDim.x) {
        output[base + col] /= total;
    }
}

__global__ void slice_kernel(const float* input, float* output,
                            std::size_t input_rows, std::size_t input_cols,
                            std::size_t row_start, std::size_t row_end,
                            std::size_t col_start, std::size_t col_end) {
    const auto row = blockIdx.y * blockDim.y + threadIdx.y;
    const auto col = blockIdx.x * blockDim.x + threadIdx.x;
    const auto out_rows = row_end - row_start;
    const auto out_cols = col_end - col_start;
    if (row < out_rows && col < out_cols) {
        output[row * out_cols + col] = input[(row_start + row) * input_cols + (col_start + col)];
    }
}

}

Matrix transpose(const Matrix& matrix) {
    Matrix output(matrix.cols(), matrix.rows());
    dim3 block(tile_size, tile_size);
    dim3 grid((matrix.cols() + tile_size - 1) / tile_size,
              (matrix.rows() + tile_size - 1) / tile_size);
    transpose_kernel<<<grid, block>>>(matrix.device_data(), output.device_data(), matrix.rows(), matrix.cols());
    checkCuda(cudaGetLastError(), "transpose kernel launch");
    return output;
}

Matrix relu(const Matrix& matrix) {
    Matrix output(matrix.rows(), matrix.cols());
    relu_kernel<<<static_cast<unsigned>((matrix.size() + 255) / 256), 256>>>(
        matrix.device_data(), output.device_data(), matrix.size());
    checkCuda(cudaGetLastError(), "relu kernel launch");
    return output;
}

Matrix softmax(const Matrix& matrix) {
    if (matrix.empty()) return Matrix(matrix.rows(), matrix.cols());
    Matrix output(matrix.rows(), matrix.cols());

    float* row_max = static_cast<float*>(allocate_device_memory(matrix.rows() * sizeof(float)));
    float* row_total = static_cast<float*>(allocate_device_memory(matrix.rows() * sizeof(float)));

    const unsigned grid_rows = static_cast<unsigned>(matrix.rows());
    softmax_row_max_kernel<<<grid_rows, softmax_block>>>(matrix.device_data(), matrix.rows(), matrix.cols(), row_max);
    softmax_row_sum_kernel<<<grid_rows, softmax_block>>>(matrix.device_data(), output.device_data(),
                                                       matrix.rows(), matrix.cols(), row_max, row_total);
    softmax_row_norm_kernel<<<grid_rows, softmax_block>>>(output.device_data(), matrix.rows(), matrix.cols(), row_total);
    checkCuda(cudaGetLastError(), "softmax kernel launch");
    free_device_memory(row_max);
    free_device_memory(row_total);
    return output;
}

Matrix flatten(const Matrix& matrix) {
    if (matrix.empty()) return Matrix(0, 1);
    Matrix output(matrix.size(), 1);
    checkCuda(cudaMemcpy(output.device_data(), matrix.device_data(), matrix.size() * sizeof(float), cudaMemcpyDeviceToDevice),
             "flatten device copy");
    return output;
}

Matrix slice(const Matrix& matrix, std::size_t row_start, std::size_t row_end,
             std::size_t col_start, std::size_t col_end) {
    if (row_start > row_end || col_start > col_end || row_end > matrix.rows() || col_end > matrix.cols()) {
        throw std::out_of_range("Invalid matrix slice");
    }
    const std::size_t out_rows = row_end - row_start;
    const std::size_t out_cols = col_end - col_start;
    Matrix output(out_rows, out_cols);
    if (output.empty()) return output;

    dim3 block(16, 16);
    dim3 grid((out_cols + block.x - 1) / block.x, (out_rows + block.y - 1) / block.y);
    slice_kernel<<<grid, block>>>(matrix.device_data(), output.device_data(),
                                 matrix.rows(), matrix.cols(),
                                 row_start, row_end, col_start, col_end);
    checkCuda(cudaGetLastError(), "slice kernel launch");
    return output;
}

Matrix Matrix::transpose() const { return matrix_pro::transpose(*this); }
Matrix Matrix::relu() const { return matrix_pro::relu(*this); }
Matrix Matrix::softmax() const { return matrix_pro::softmax(*this); }
Matrix Matrix::flatten() const { return matrix_pro::flatten(*this); }
Matrix Matrix::slice(std::size_t row_start, std::size_t row_end, std::size_t col_start, std::size_t col_end) const {
    return matrix_pro::slice(*this, row_start, row_end, col_start, col_end);
}

}