#include "matrix_pro/operations.hpp"
#include "matrix_pro/cuda_utils.hpp"

#include <stdexcept>

namespace matrix_pro {
namespace {

constexpr unsigned tile_size = 16;

__global__ void matmul_kernel(const float* left, const float* right, float* output,
                              std::size_t left_rows, std::size_t shared, std::size_t right_cols) {
    __shared__ float left_tile[tile_size][tile_size];
    __shared__ float right_tile[tile_size][tile_size];
    const auto row = blockIdx.y * tile_size + threadIdx.y;
    const auto col = blockIdx.x * tile_size + threadIdx.x;
    float value = 0.0f;

    for (std::size_t tile = 0; tile < shared; tile += tile_size) {
        const auto left_col = tile + threadIdx.x;
        const auto right_row = tile + threadIdx.y;
        left_tile[threadIdx.y][threadIdx.x] = row < left_rows && left_col < shared
            ? left[row * shared + left_col] : 0.0f;
        right_tile[threadIdx.y][threadIdx.x] = right_row < shared && col < right_cols
            ? right[right_row * right_cols + col] : 0.0f;
        __syncthreads();
        for (unsigned index = 0; index < tile_size; ++index) {
            value += left_tile[threadIdx.y][index] * right_tile[index][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < left_rows && col < right_cols) output[row * right_cols + col] = value;
}

}

Matrix multiply(const Matrix& left, const Matrix& right) {
    if (left.cols() != right.rows()) {
        throw std::invalid_argument("Matrix dimensions are incompatible for multiplication");
    }
    Matrix output(left.rows(), right.cols());
    dim3 block(tile_size, tile_size);
    dim3 grid((right.cols() + tile_size - 1) / tile_size,
              (left.rows() + tile_size - 1) / tile_size);
    matmul_kernel<<<grid, block>>>(left.device_data(), right.device_data(), output.device_data(),
                                   left.rows(), left.cols(), right.cols());
    checkCuda(cudaGetLastError(), "matrix multiplication kernel launch");
    return output;
}

}
