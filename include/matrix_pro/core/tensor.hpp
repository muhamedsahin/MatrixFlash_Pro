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
    const std::vector<std::size_t>& strides() const noexcept { return strides_; }
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

    // ---- stride-aware N-D operations (Phase-2 Tensor Engine) ----

    // True when the strides are contiguous row-major (no gaps, no permutation).
    bool is_contiguous() const noexcept;

    // Returns a contiguous copy if the tensor is non-contiguous, or *this if
    // already contiguous.
    Tensor contiguous() const;

    // Zero-copy dimension reordering.  E.g. permute({0,3,1,2}) for NHWC→NCHW.
    Tensor permute(const std::vector<std::size_t>& dims) const;

    // Zero-copy broadcast expansion along size-1 dimensions (stride set to 0).
    Tensor expand(const std::vector<std::size_t>& new_shape) const;

    // Reshape (contiguous tensors only — throws if non-contiguous).
    Tensor view(const std::vector<std::size_t>& new_shape) const;

    // Remove a size-1 dimension.
    Tensor squeeze(std::size_t dim) const;

    // Insert a size-1 dimension.
    Tensor unsqueeze(std::size_t dim) const;

    // Narrow along a given dimension (zero-copy slice).
    Tensor narrow(std::size_t dim, std::size_t start, std::size_t length) const;

    // Select a single index along a dimension, reducing rank by one.
    Tensor select(std::size_t dim, std::size_t index) const;

    // Element offset helper for N-D indexing.
    std::size_t storage_offset() const noexcept { return storage_offset_; }

private:
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> strides_;
    std::size_t size_ = 0;
    std::size_t storage_offset_ = 0;
    std::vector<float> host_data_;
    std::unique_ptr<float, void (*)(float*)> device_data_{nullptr, nullptr};
    bool device_only_ = false;
    // Same fail-fast stale-mirror contract as Matrix::host_current_.
    bool host_current_ = true;
    // True when this tensor shares storage with another (e.g. after permute/narrow).
    bool is_view_ = false;

    void allocate_device();
    void ensure_host_buffer();
    static std::vector<std::size_t> compute_contiguous_strides(const std::vector<std::size_t>& shape);
};

}