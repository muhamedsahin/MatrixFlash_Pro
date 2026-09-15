#include "matrix_pro/operations.hpp"
#include "matrix_pro/cuda_utils.hpp"

#include <cfloat>
#include <cmath>
#include <stdexcept>

namespace matrix_pro {
namespace {

constexpr unsigned tile_size = 16;

__global__ void transpose_kernel(const float* input, float* output, std::size_t rows, std::size_t cols) {
    const auto col = blockIdx.x * blockDim.x + threadIdx.x;
    const auto row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && col < cols) output[col * rows + row] = input[row * cols + col];
}

__global__ void relu_kernel(const float* input, float* output, std::size_t count) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) output[index] = input[index] > 0.0f ? input[index] : 0.0f;
}

__global__ void softmax_kernel(const float* input, float* output, std::size_t count) {
    __shared__ float shared_max;
    __shared__ float shared_total;
    if (threadIdx.x == 0) {
        float maximum = -FLT_MAX;
        for (std::size_t index = 0; index < count; ++index) maximum = fmaxf(maximum, input[index]);
        shared_max = maximum;
        float total = 0.0f;
        for (std::size_t index = 0; index < count; ++index) total += expf(input[index] - maximum);
        shared_total = total;
    }
    __syncthreads();
    for (std::size_t index = threadIdx.x; index < count; index += blockDim.x) output[index] = expf(input[index] - shared_max) / shared_total;
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
    relu_kernel<<<static_cast<unsigned>((matrix.size() + 255) / 256), 256>>>(matrix.device_data(), output.device_data(), matrix.size());
    checkCuda(cudaGetLastError(), "relu kernel launch");
    return output;
}

Matrix softmax(const Matrix& matrix) {
    Matrix output(matrix.rows(), matrix.cols());
    softmax_kernel<<<1, 256>>>(matrix.device_data(), output.device_data(), matrix.size());
    checkCuda(cudaGetLastError(), "softmax kernel launch");
    return output;
}

Matrix flatten(const Matrix& matrix) {
    Matrix copy = matrix;
    copy.download();
    Matrix output(matrix.size(), 1, copy.data());
    return output;
}

Matrix slice(const Matrix& matrix, std::size_t row_start, std::size_t row_end,
             std::size_t col_start, std::size_t col_end) {
    if (row_start > row_end || col_start > col_end || row_end > matrix.rows() || col_end > matrix.cols()) {
        throw std::out_of_range("Invalid matrix slice");
    }
    Matrix copy = matrix;
    copy.download();
    std::vector<float> values;
    values.reserve((row_end - row_start) * (col_end - col_start));
    for (std::size_t row = row_start; row < row_end; ++row) {
        for (std::size_t col = col_start; col < col_end; ++col) values.push_back(copy.at(row, col));
    }
    return Matrix(row_end - row_start, col_end - col_start, values);
}

Matrix Matrix::transpose() const { return matrix_pro::transpose(*this); }
Matrix Matrix::relu() const { return matrix_pro::relu(*this); }
Matrix Matrix::softmax() const { return matrix_pro::softmax(*this); }
Matrix Matrix::flatten() const { return matrix_pro::flatten(*this); }
Matrix Matrix::slice(std::size_t row_start, std::size_t row_end, std::size_t col_start, std::size_t col_end) const {
    return matrix_pro::slice(*this, row_start, row_end, col_start, col_end);
}

}
