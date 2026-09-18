#include "matrix_pro/core/matrix.hpp"
#include "matrix_pro/core/cuda_utils.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <utility>

namespace matrix_pro {
namespace {

void release_device(float* pointer) {
    if (pointer != nullptr) matrix_pro::free_device_memory(pointer);
}

}

Matrix::Matrix(std::size_t rows, std::size_t cols)
    : Matrix(rows, cols, MemoryMode::host_and_device) {}

Matrix::Matrix(std::size_t rows, std::size_t cols, MemoryMode mode)
    : rows_(rows), cols_(cols), device_data_(nullptr, release_device),
      device_only_(mode == MemoryMode::device_only) {
    if (!device_only_) host_data_.resize(rows * cols);
    allocate_device();
}

Matrix::Matrix(std::size_t rows, std::size_t cols, const std::vector<float>& values)
    : Matrix(rows, cols) {
    if (values.size() != size()) throw ShapeMismatchError("Matrix data size does not match shape");
    host_data_ = values;
    upload();
}

Matrix::Matrix(std::initializer_list<std::initializer_list<float>> values)
    : Matrix(values.size(), values.size() == 0 ? 0 : values.begin()->size()) {
    std::size_t row_index = 0;
    for (const auto& row : values) {
        if (row.size() != cols_) throw ShapeMismatchError("All matrix rows must have the same size");
        std::copy(row.begin(), row.end(), host_data_.begin() + row_index * cols_);
        ++row_index;
    }
    upload();
}

Matrix::Matrix(const Matrix& other)
    : rows_(other.rows_), cols_(other.cols_), host_data_(other.host_data_),
      device_data_(nullptr, release_device), device_only_(other.device_only_),
      host_current_(other.host_current_) {
    allocate_device();
    if (size() != 0) {
        checkCuda(cudaMemcpyAsync(device_data_.get(), other.device_data_.get(), size() * sizeof(float), cudaMemcpyDeviceToDevice, compute_stream()), "cudaMemcpy device to device");
    }
}

Matrix::~Matrix() {
    release_host_pinned();
}

Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) *this = Matrix(other);
    return *this;
}

Matrix::Matrix(Matrix&& other) noexcept
    : rows_(other.rows_), cols_(other.cols_),
      host_data_(std::move(other.host_data_)),
      device_data_(std::move(other.device_data_)),
      host_pinned_(other.host_pinned_),
      device_only_(other.device_only_),
      host_current_(other.host_current_) {
    other.rows_ = 0;
    other.cols_ = 0;
    other.host_pinned_ = false;
    other.device_only_ = false;
    other.host_current_ = true;
}

Matrix& Matrix::operator=(Matrix&& other) noexcept {
    if (this != &other) {
        // The destination may currently hold a host buffer pinned with
        // cudaHostRegister. Unregister it before releasing so the address can be
        // safely re-registered by a later allocation.
        release_host_pinned();
        rows_ = other.rows_;
        cols_ = other.cols_;
        host_data_ = std::move(other.host_data_);
        device_data_ = std::move(other.device_data_);
        host_pinned_ = other.host_pinned_;
        device_only_ = other.device_only_;
        host_current_ = other.host_current_;
        other.rows_ = 0;
        other.cols_ = 0;
        other.host_pinned_ = false;
        other.device_only_ = false;
        other.host_current_ = true;
    }
    return *this;
}

void Matrix::allocate_device() {
    if (size() == 0) return;
    void* pointer = matrix_pro::allocate_device_memory(size() * sizeof(float));
    device_data_.reset(static_cast<float*>(pointer));
}

void Matrix::ensure_host_pinned() {
    if (host_pinned_ || host_data_.empty()) return;
    const cudaError_t status = cudaHostRegister(host_data_.data(), host_data_.size() * sizeof(float), cudaHostRegisterDefault);
    if (status == cudaSuccess) {
        host_pinned_ = true;
    } else if (status != cudaErrorHostMemoryAlreadyRegistered) {
        throw MatrixProError(std::string("cudaHostRegister failed: ") + cudaGetErrorString(status));
    } else {
        host_pinned_ = true;
    }
}

void Matrix::release_host_pinned() {
    if (!host_pinned_ || host_data_.empty()) return;
    const cudaError_t status = cudaHostUnregister(host_data_.data());
    if (status != cudaSuccess && status != cudaErrorHostMemoryNotRegistered) {
        throw MatrixProError(std::string("cudaHostUnregister failed: ") + cudaGetErrorString(status));
    }
    host_pinned_ = false;
}

void Matrix::ensure_host_buffer() {
    if (host_data_.size() != size()) host_data_.resize(size());
}

void Matrix::upload() {
    if (size() == 0) return;
    // In device_only mode the device buffer is authoritative; there is nothing
    // to push and the host mirror may not even exist.
    if (device_only_) return;
    checkCuda(cudaMemcpyAsync(device_data_.get(), host_data_.data(), size() * sizeof(float), cudaMemcpyHostToDevice, compute_stream()), "cudaMemcpy host to device");
    checkCuda(cudaStreamSynchronize(compute_stream()), "cudaStreamSynchronize after upload");
    mark_host_current();
}

void Matrix::download() {
    if (size() == 0) return;
    ensure_host_buffer();
    checkCuda(cudaMemcpyAsync(host_data_.data(), device_data_.get(), size() * sizeof(float), cudaMemcpyDeviceToHost, compute_stream()), "cudaMemcpy device to host");
    checkCuda(cudaStreamSynchronize(compute_stream()), "cudaStreamSynchronize after download");
    mark_host_current();
}

void Matrix::require_host_current(const char* accessor) const {
    if (!host_materialized()) {
        throw InvalidArgumentError(std::string(accessor) + " has no host data; call download() first for a device-only matrix");
    }
    if (!host_current_) {
        throw InvalidArgumentError(std::string(accessor) + " host mirror is stale (device holds newer data); call download() first");
    }
}

float& Matrix::at(std::size_t row, std::size_t col) {
    validate_index(row, col);
    require_host_current("Matrix::at()");
    return host_data_[row * cols_ + col];
}

const float& Matrix::at(std::size_t row, std::size_t col) const {
    validate_index(row, col);
    require_host_current("Matrix::at()");
    return host_data_[row * cols_ + col];
}

const std::vector<float>& Matrix::data() const {
    require_host_current("Matrix::data()");
    return host_data_;
}

void Matrix::synchronize() const {
    matrix_pro::synchronize();
}

void Matrix::validate_index(std::size_t row, std::size_t col) const {
    if (row >= rows_ || col >= cols_) throw OutOfRangeError("Matrix index out of range");
}

void Matrix::validate_same_shape(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) throw ShapeMismatchError("Matrix shapes must match");
}

void Matrix::save(const std::string& filename) const {
    Matrix copy = *this;
    copy.download();
    std::ofstream file(filename, std::ios::binary);
    if (!file) throw IoError("Could not open file for writing: " + filename);
    // Versioned container: magic + version + header + payload.
    constexpr char magic[4] = {'M', 'F', 'M', 'P'};
    constexpr std::uint32_t version = 1;
    file.write(magic, sizeof(magic));
    file.write(reinterpret_cast<const char*>(&version), sizeof(version));
    const std::size_t rows = rows_, cols = cols_;
    file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    file.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    file.write(reinterpret_cast<const char*>(copy.data().data()), static_cast<std::streamsize>(size() * sizeof(float)));
    if (!file) throw IoError("Failed to write matrix file: " + filename);
}

Matrix Matrix::load(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw IoError("Could not open file for reading: " + filename);
    char magic[4] = {};
    file.read(magic, sizeof(magic));
    const bool versioned = magic[0] == 'M' && magic[1] == 'F' && magic[2] == 'M' && magic[3] == 'P';
    if (versioned) {
        std::uint32_t version = 0;
        file.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (version != 1) throw IoError("Unsupported matrix file version: " + std::to_string(version));
    } else if (file.tellg() >= 0) {
        // Legacy (pre-versioning) format: rewind and read the plain header.
        file.clear();
        file.seekg(0, std::ios::beg);
    }
    std::size_t rows = 0, cols = 0;
    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    if (!file) throw IoError("Corrupt matrix header in file: " + filename);
    std::vector<float> values(rows * cols);
    file.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(rows * cols * sizeof(float)));
    if (!file) throw IoError("Truncated matrix data in file: " + filename);
    return Matrix(rows, cols, values);
}

}

