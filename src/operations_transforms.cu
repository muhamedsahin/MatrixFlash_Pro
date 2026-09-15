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

__global__ void softmax_init_kernel(float* global_max, float* global_total) {
    if (threadIdx.x == 0) {
        *global_max = -FLT_MAX;
        *global_total = 0.0f;
    }
}

__device__ float atomicMaxf(float* address, float value) {
    int* address_as_int = reinterpret_cast<int*>(address);
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                        __float_as_int(fmaxf(value, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void transpose_kernel(const float* input, float* output, std::size_t rows, std::size_t cols) {
    const auto col = blockIdx.x * blockDim.x + threadIdx.x;
    const auto row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && col < cols) output[col * rows + row] = input[row * cols + col];
}

__global__ void relu_kernel(const float* input, float* output, std::size_t count) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) output[index] = input[index] > 0.0f ? input[index] : 0.0f;
}

__global__ void softmax_max_kernel(const float* input, std::size_t count, float* global_max) {
    __shared__ float shared[softmax_block];
    const auto tid = threadIdx.x;
    float local = -FLT_MAX;
    for (std::size_t index = blockIdx.x * blockDim.x + tid; index < count; index += gridDim.x * blockDim.x) {
        local = fmaxf(local, input[index]);
    }
    shared[tid] = local;
    __syncthreads();
    for (unsigned stride = softmax_block / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        __syncthreads();
    }
    if (tid == 0) atomicMaxf(global_max, shared[0]);
}

__global__ void softmax_sum_kernel(const float* input, float* output, std::size_t count,
                                   float* global_max, float* global_total) {
    __shared__ float shared[softmax_block];
    const auto tid = threadIdx.x;
    float local = 0.0f;
    const float maximum = *global_max;
    for (std::size_t index = blockIdx.x * blockDim.x + tid; index < count; index += gridDim.x * blockDim.x) {
        const float value = expf(input[index] - maximum);
        output[index] = value;
        local += value;
    }
    shared[tid] = local;
    __syncthreads();
    for (unsigned stride = softmax_block / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared[tid] += shared[tid + stride];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(global_total, shared[0]);
}

__global__ void softmax_norm_kernel(float* output, std::size_t count, const float* global_total) {
    const float total = *global_total;
    for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count; index += gridDim.x * blockDim.x) {
        output[index] /= total;
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
    const std::size_t count = matrix.size();
    if (count == 0) return Matrix(matrix.rows(), matrix.cols());
    Matrix output(matrix.rows(), matrix.cols());

    float* global_max = nullptr;
    float* global_total = nullptr;
    checkCuda(cudaMalloc(&global_max, sizeof(float)), "softmax alloc max");
    checkCuda(cudaMalloc(&global_total, sizeof(float)), "softmax alloc total");
    softmax_init_kernel<<<1, 32>>>(global_max, global_total);
    checkCuda(cudaGetLastError(), "softmax init launch");

    const unsigned grid = static_cast<unsigned>((count + softmax_block - 1) / softmax_block);
    softmax_max_kernel<<<grid, softmax_block>>>(matrix.device_data(), count, global_max);
    softmax_sum_kernel<<<grid, softmax_block>>>(matrix.device_data(), output.device_data(), count, global_max, global_total);
    softmax_norm_kernel<<<grid, softmax_block>>>(output.device_data(), count, global_total);
    checkCuda(cudaGetLastError(), "softmax kernel launch");
    checkCuda(cudaFree(global_max), "softmax free max");
    checkCuda(cudaFree(global_total), "softmax free total");
    return output;
}

Matrix flatten(const Matrix& matrix) {
    Matrix copy = matrix;
    copy.download();
    std::vector<float> values(copy.data().begin(), copy.data().end());
    if (matrix.empty()) return Matrix(0, 1);
    return Matrix(matrix.size(), 1, values);
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