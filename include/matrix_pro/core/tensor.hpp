#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "matrix_pro/core/errors.hpp"
#include "matrix_pro/core/memory_mode.hpp"

namespace matrix_pro {

class Tensor {
public:
    Tensor() = default;
    explicit Tensor(std::vector<std::size_t> shape);
    // device_only skips the host mirror until download()/host access is needed.
    Tensor(std::vector<std::size_t> shape, MemoryMode mode);
    Tensor(std::vector<std::size_t> shape, const std::vector<float>& values);
    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;
    ~Tensor();

    std::size_t rank() const noexcept { return shape_.size(); }
    const std::vector<std::size_t>& shape() const noexcept { return shape_; }
    std::size_t size() const noexcept { return size_; }
    bool empty() const noexcept { return size_ == 0; }
    bool device_only() const noexcept { return device_only_; }
    bool host_materialized() const noexcept { return host_data_.size() == size_; }
    // True when the host mirror reflects the current device contents (see
    // Matrix::host_current for the full contract).
    bool host_current() const noexcept { return host_current_; }
    void mark_host_stale() noexcept { host_current_ = false; }
    void mark_host_current() noexcept { host_current_ = true; }
    const std::vector<float>& data() const;
    std::vector<float>& data() noexcept { return host_data_; }
    const float* host_ptr() const noexcept { return host_data_.data(); }
    float* host_ptr() noexcept { return host_data_.data(); }
    float* device_data() noexcept { return device_data_.get(); }
    const float* device_data() const noexcept { return device_data_.get(); }

    void upload();
    void download();
    void fill(float value);
    // Clears both the host mirror and the device buffer without a host round-trip.
    // Gradient accumulators must start at zero because backward passes accumulate.
    void zero();
    void synchronize() const;

private:
    std::vector<std::size_t> shape_;
    std::size_t size_ = 0;
    std::vector<float> host_data_;
    std::unique_ptr<float, void (*)(float*)> device_data_{nullptr, nullptr};
    bool device_only_ = false;
    // Same fail-fast stale-mirror contract as Matrix::host_current_.
    bool host_current_ = true;

    void allocate_device();
    void ensure_host_buffer();
};

}