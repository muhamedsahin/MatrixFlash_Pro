#include "matrix_pro/matrix.hpp"
#include "matrix_pro/cuda_utils.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace matrix_pro {
namespace {

void release_device(float* pointer) {
    if (pointer != nullptr) cudaFree(pointer);
}

}

Matrix::Matrix(std::size_t rows, std::size_t cols)
    : rows_(rows), cols_(cols), host_data_(rows * cols), device_data_(nullptr, release_device) {
    allocate_device();
    upload();
}

Matrix::Matrix(std::size_t rows, std::size_t cols, const std::vector<float>& values)
    : Matrix(rows, cols) {
    if (values.size() != size()) throw std::invalid_argument("Matrix data size does not match shape");
    host_data_ = values;
    upload();
}

Matrix::Matrix(std::initializer_list<std::initializer_list<float>> values)
    : Matrix(values.size(), values.size() == 0 ? 0 : values.begin()->size()) {
    std::size_t row_index = 0;
    for (const auto& row : values) {
        if (row.size() != cols_) throw std::invalid_argument("All matrix rows must have the same size");
        std::copy(row.begin(), row.end(), host_data_.begin() + row_index * cols_);
        ++row_index;
    }
    upload();
}

Matrix::Matrix(const Matrix& other)
    : Matrix(other.rows_, other.cols_) {
    if (size() != 0) {
        checkCuda(cudaMemcpy(device_data_.get(), other.device_data_.get(), size() * sizeof(float), cudaMemcpyDeviceToDevice), "cudaMemcpy device to device");
    }
    host_data_ = other.host_data_;
}

Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) *this = Matrix(other);
    return *this;
}

void Matrix::allocate_device() {
    if (size() == 0) return;
    float* pointer = nullptr;
    checkCuda(cudaMalloc(&pointer, size() * sizeof(float)), "cudaMalloc");
    device_data_.reset(pointer);
}

void Matrix::upload() {
    if (size() == 0) return;
    checkCuda(cudaMemcpy(device_data_.get(), host_data_.data(), size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy host to device");
}

void Matrix::download() {
    if (size() == 0) return;
    checkCuda(cudaMemcpy(host_data_.data(), device_data_.get(), size() * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy device to host");
}

void Matrix::synchronize() const {
    matrix_pro::synchronize();
}

void Matrix::validate_index(std::size_t row, std::size_t col) const {
    if (row >= rows_ || col >= cols_) throw std::out_of_range("Matrix index out of range");
}

float& Matrix::at(std::size_t row, std::size_t col) {
    validate_index(row, col);
    return host_data_[row * cols_ + col];
}

const float& Matrix::at(std::size_t row, std::size_t col) const {
    validate_index(row, col);
    return host_data_[row * cols_ + col];
}

void Matrix::validate_same_shape(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) throw std::invalid_argument("Matrix shapes must match");
}

void Matrix::save(const std::string& filename) const {
    Matrix copy = *this;
    copy.download();
    std::ofstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Could not open file for writing: " + filename);
    const std::size_t rows = rows_, cols = cols_;
    file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    file.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    file.write(reinterpret_cast<const char*>(copy.data().data()), static_cast<std::streamsize>(size() * sizeof(float)));
    if (!file) throw std::runtime_error("Failed to write matrix file: " + filename);
}

Matrix Matrix::load(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Could not open file for reading: " + filename);
    std::size_t rows = 0, cols = 0;
    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    if (!file) throw std::runtime_error("Corrupt matrix header in file: " + filename);
    std::vector<float> values(rows * cols);
    file.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(rows * cols * sizeof(float)));
    if (!file) throw std::runtime_error("Truncated matrix data in file: " + filename);
    return Matrix(rows, cols, values);
}

}

