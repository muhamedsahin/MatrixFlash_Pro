#include "matrix_pro/matrix.hpp"
#include "matrix_pro/cuda_utils.hpp"

#include <algorithm>
#include <stdexcept>

namespace matrix_pro {
namespace {

__global__ void fill_kernel(float* values, std::size_t count, float value) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) values[index] = value;
}

}

void Matrix::fill(float value) {
    if (size() == 0) return;
    fill_kernel<<<static_cast<unsigned>((size() + 255) / 256), 256>>>(device_data_.get(), size(), value);
    checkCuda(cudaGetLastError(), "fill kernel launch");
    synchronize();
    std::fill(host_data_.begin(), host_data_.end(), value);
}

Matrix Matrix::zeros(std::size_t rows, std::size_t cols) {
    return Matrix(rows, cols);
}

Matrix Matrix::ones(std::size_t rows, std::size_t cols) {
    Matrix result(rows, cols);
    result.fill(1.0f);
    return result;
}

Matrix Matrix::identity(std::size_t size) {
    Matrix result(size, size);
    for (std::size_t index = 0; index < size; ++index) result.at(index, index) = 1.0f;
    result.upload();
    return result;
}

}
