#include "matrix_pro/core/matrix.hpp"
#include "matrix_pro/core/cuda_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
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
    if (host_materialized()) {
        // Host-and-device mode: write the mirror, then push it over once.
        // upload() synchronizes and marks the mirror current.
        std::fill(host_data_.begin(), host_data_.end(), value);
        upload();
    } else {
        // device_only with no host mirror yet: write straight to the device.
        fill_kernel<<<static_cast<unsigned>((size() + 255) / 256), 256, 0, compute_stream()>>>(device_data_.get(), size(), value);
        checkCuda(cudaGetLastError(), "fill kernel launch");
    }
}

Matrix Matrix::zeros(std::size_t rows, std::size_t cols) {
    Matrix result(rows, cols);
    if (result.size() != 0) {
        checkCuda(cudaMemsetAsync(result.device_data_.get(), 0, result.size() * sizeof(float), compute_stream()), "cudaMemset for zero matrix");
    }
    return result;
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

Matrix Matrix::random(std::size_t rows, std::size_t cols) {
    return Matrix::random(rows, cols, std::random_device{}());
}

Matrix Matrix::random(std::size_t rows, std::size_t cols, std::uint32_t seed) {
    Matrix result(rows, cols);
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    for (auto& value : result.host_data_) value = distribution(generator);
    result.upload();
    return result;
}

Matrix Matrix::uniform(std::size_t rows, std::size_t cols, float low, float high) {
    return Matrix::uniform(rows, cols, low, high, std::random_device{}());
}

Matrix Matrix::uniform(std::size_t rows, std::size_t cols, float low, float high, std::uint32_t seed) {
    Matrix result(rows, cols);
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(low, high);
    for (auto& value : result.host_data_) value = distribution(generator);
    result.upload();
    return result;
}

Matrix Matrix::randn(std::size_t rows, std::size_t cols) {
    return Matrix::randn(rows, cols, std::random_device{}());
}

Matrix Matrix::randn(std::size_t rows, std::size_t cols, std::uint32_t seed) {
    Matrix result(rows, cols);
    std::mt19937 generator(seed);
    std::normal_distribution<float> distribution(0.0f, 1.0f);
    for (auto& value : result.host_data_) value = distribution(generator);
    result.upload();
    return result;
}

Matrix Matrix::glorot(std::size_t rows, std::size_t cols) {
    const float limit = std::sqrt(6.0f / static_cast<float>(rows + cols));
    return Matrix::uniform(rows, cols, -limit, limit);
}

}

