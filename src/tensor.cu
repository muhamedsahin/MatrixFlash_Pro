#include "matrix_pro/tensor.hpp"
#include "matrix_pro/cuda_utils.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace matrix_pro {
namespace {
void release_tensor_device(float* pointer) {
    if (pointer != nullptr) free_device_memory(pointer);
}
}

Tensor::Tensor(std::vector<std::size_t> shape)
    : shape_(std::move(shape)), device_data_(nullptr, release_tensor_device) {
    size_ = shape_.empty() ? 0 : std::accumulate(shape_.begin(), shape_.end(), std::size_t{1}, std::multiplies<>());
    host_data_.resize(size_);
    allocate_device();
}

Tensor::Tensor(std::vector<std::size_t> shape, const std::vector<float>& values)
    : Tensor(std::move(shape)) {
    if (values.size() != size_) throw std::invalid_argument("Tensor data size does not match shape");
    host_data_ = values;
    upload();
}

Tensor::Tensor(const Tensor& other)
    : shape_(other.shape_), size_(other.size_), host_data_(other.host_data_), device_data_(nullptr, release_tensor_device) {
    allocate_device();
    if (size_ != 0) checkCuda(cudaMemcpyAsync(device_data_.get(), other.device_data_.get(), size_ * sizeof(float), cudaMemcpyDeviceToDevice, compute_stream()), "Tensor device copy");
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) *this = Tensor(other);
    return *this;
}

Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)), size_(other.size_), host_data_(std::move(other.host_data_)), device_data_(std::move(other.device_data_)) {
    other.size_ = 0;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        shape_ = std::move(other.shape_);
        size_ = other.size_;
        host_data_ = std::move(other.host_data_);
        device_data_ = std::move(other.device_data_);
        other.size_ = 0;
    }
    return *this;
}

Tensor::~Tensor() = default;

void Tensor::allocate_device() {
    if (size_ != 0) device_data_.reset(static_cast<float*>(allocate_device_memory(size_ * sizeof(float))));
}

void Tensor::upload() {
    if (size_ != 0) {
        checkCuda(cudaMemcpyAsync(device_data_.get(), host_data_.data(), size_ * sizeof(float), cudaMemcpyHostToDevice, compute_stream()), "Tensor upload");
        checkCuda(cudaStreamSynchronize(compute_stream()), "Tensor upload synchronize");
    }
}

void Tensor::download() {
    if (size_ != 0) {
        checkCuda(cudaMemcpyAsync(host_data_.data(), device_data_.get(), size_ * sizeof(float), cudaMemcpyDeviceToHost, compute_stream()), "Tensor download");
        checkCuda(cudaStreamSynchronize(compute_stream()), "Tensor download synchronize");
    }
}

void Tensor::fill(float value) {
    std::fill(host_data_.begin(), host_data_.end(), value);
    upload();
}

void Tensor::synchronize() const { matrix_pro::synchronize(); }
}