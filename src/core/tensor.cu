#include "matrix_pro/core/tensor.hpp"
#include "matrix_pro/core/cuda_utils.hpp"

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

__global__ void tensor_fill_kernel(float* values, std::size_t count, float value) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) values[index] = value;
}
}

Tensor::Tensor(std::vector<std::size_t> shape)
    : Tensor(std::move(shape), MemoryMode::host_and_device) {}

Tensor::Tensor(std::vector<std::size_t> shape, MemoryMode mode)
    : shape_(std::move(shape)), device_data_(nullptr, release_tensor_device),
      device_only_(mode == MemoryMode::device_only) {
    size_ = shape_.empty() ? 0 : std::accumulate(shape_.begin(), shape_.end(), std::size_t{1}, std::multiplies<>());
    if (!device_only_) host_data_.resize(size_);
    allocate_device();
}

Tensor::Tensor(std::vector<std::size_t> shape, const std::vector<float>& values)
    : Tensor(std::move(shape)) {
    if (values.size() != size_) throw ShapeMismatchError("Tensor data size does not match shape");
    host_data_ = values;
    upload();
}

Tensor::Tensor(const Tensor& other)
    : shape_(other.shape_), size_(other.size_), host_data_(other.host_data_),
      device_data_(nullptr, release_tensor_device), device_only_(other.device_only_),
      host_current_(other.host_current_) {
    allocate_device();
    if (size_ != 0) checkCuda(cudaMemcpyAsync(device_data_.get(), other.device_data_.get(), size_ * sizeof(float), cudaMemcpyDeviceToDevice, compute_stream()), "Tensor device copy");
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) *this = Tensor(other);
    return *this;
}

Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)), size_(other.size_), host_data_(std::move(other.host_data_)),
      device_data_(std::move(other.device_data_)), device_only_(other.device_only_),
      host_current_(other.host_current_) {
    other.size_ = 0;
    other.device_only_ = false;
    other.host_current_ = true;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        shape_ = std::move(other.shape_);
        size_ = other.size_;
        host_data_ = std::move(other.host_data_);
        device_data_ = std::move(other.device_data_);
        device_only_ = other.device_only_;
        host_current_ = other.host_current_;
        other.size_ = 0;
        other.device_only_ = false;
        other.host_current_ = true;
    }
    return *this;
}

Tensor::~Tensor() = default;

void Tensor::allocate_device() {
    if (size_ != 0) device_data_.reset(static_cast<float*>(allocate_device_memory(size_ * sizeof(float))));
}

void Tensor::ensure_host_buffer() {
    if (host_data_.size() != size_) host_data_.resize(size_);
}

void Tensor::upload() {
    if (size_ != 0 && !device_only_) {
        checkCuda(cudaMemcpyAsync(device_data_.get(), host_data_.data(), size_ * sizeof(float), cudaMemcpyHostToDevice, compute_stream()), "Tensor upload");
        checkCuda(cudaStreamSynchronize(compute_stream()), "Tensor upload synchronize");
        mark_host_current();
    }
}

void Tensor::download() {
    if (size_ != 0) {
        ensure_host_buffer();
        checkCuda(cudaMemcpyAsync(host_data_.data(), device_data_.get(), size_ * sizeof(float), cudaMemcpyDeviceToHost, compute_stream()), "Tensor download");
        checkCuda(cudaStreamSynchronize(compute_stream()), "Tensor download synchronize");
        mark_host_current();
    }
}

const std::vector<float>& Tensor::data() const {
    if (host_data_.size() != size_)
        throw InvalidArgumentError("Tensor::data() has no host data; call download() first");
    if (!host_current_)
        throw InvalidArgumentError("Tensor::data() host mirror is stale (device holds newer data); call download() first");
    return host_data_;
}

void Tensor::fill(float value) {
    if (size_ == 0) return;
    if (host_materialized()) {
        std::fill(host_data_.begin(), host_data_.end(), value);
        upload();
    } else {
        // device_only with no host mirror yet: write straight to the device.
        tensor_fill_kernel<<<static_cast<unsigned>((size_ + 255) / 256), 256, 0, compute_stream()>>>(
            device_data_.get(), size_, value);
        checkCuda(cudaGetLastError(), "Tensor fill kernel launch");
    }
}

void Tensor::zero() {
    if (!host_data_.empty()) std::fill(host_data_.begin(), host_data_.end(), 0.0f);
    if (size_ != 0) zero_device_memory(device_data_.get(), size_ * sizeof(float));
    // Both mirrors now hold zeros: the mirror is current again.
    if (host_materialized()) mark_host_current();
}

void Tensor::synchronize() const { matrix_pro::synchronize(); }
}