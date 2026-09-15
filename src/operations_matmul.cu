#include "matrix_pro/operations.hpp"
#include "matrix_pro/cuda_utils.hpp"

#include <cublas_v2.h>
#include <stdexcept>

namespace matrix_pro {
namespace {

// Fast, dependency-free tile kernel used as a fallback reference.
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

cublasHandle_t& cublas_handle() {
    static cublasHandle_t handle = nullptr;
    if (handle == nullptr) {
        if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cuBLAS handle creation failed");
        }
    }
    return handle;
}

Matrix multiply_tiled(const Matrix& left, const Matrix& right) {
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

__global__ void outer_product_kernel(const float* left, const float* right, float* output,
                                     std::size_t left_count, std::size_t right_count) {
    const auto col = blockIdx.x * blockDim.x + threadIdx.x;
    const auto row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < left_count && col < right_count) output[row * right_count + col] = left[row] * right[col];
}
}

Matrix multiply(const Matrix& left, const Matrix& right) {
    if (left.cols() != right.rows()) {
        throw std::invalid_argument("Matrix dimensions are incompatible for multiplication");
    }
    if (left.rows() * right.cols() == 0) return Matrix(left.rows(), right.cols());

    const int m = left.rows();
    const int n = right.cols();
    const int k = left.cols();
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // cuBLAS is column-major. For row-major C = A*B (A:MxK, B:KxN) we use the
    // standard layout trick: the column-major result (N x M) written by
    // cublasSgemm lands exactly at our row-major (M x N) buffer positions.
    Matrix output(left.rows(), right.cols());
    const cublasStatus_t status = cublasSgemm(
        cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
        n, m, k,
        &alpha,
        right.device_data(), n,
        left.device_data(), k,
        &beta,
        output.device_data(), n);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS sgemm failed");
    }
    return output;
}

Matrix Matrix::operator*(const Matrix& other) const { return matrix_pro::multiply(*this, other); }
Matrix Matrix::outer_product(const Matrix& other) const { return matrix_pro::outer_product(*this, other); }

Matrix outer_product(const Matrix& left, const Matrix& right) {
    if (left.size() == 0 || right.size() == 0) return Matrix(left.size(), right.size());
    const std::size_t lc = left.size();
    const std::size_t rc = right.size();
    Matrix output(lc, rc);
    dim3 block(16, 16);
    dim3 grid((rc + 15) / 16, (lc + 15) / 16);
    outer_product_kernel<<<grid, block>>>(left.device_data(), right.device_data(), output.device_data(), lc, rc);
    checkCuda(cudaGetLastError(), "outer product kernel launch");
    return output;
}

}