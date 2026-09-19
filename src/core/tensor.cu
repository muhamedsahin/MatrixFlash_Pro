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

// Shared-storage view deleter — does NOT free the underlying buffer.
void noop_deleter(float*) {}

__global__ void tensor_fill_kernel(float* values, std::size_t count, float value) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) values[index] = value;
}

// Generic N-D strided copy: src[multi_index ⋅ src_strides + src_offset] → dst[linear]
__global__ void strided_copy_kernel(const float* __restrict__ src,
                                     float*       __restrict__ dst,
                                     const std::size_t* __restrict__ shape,
                                     const std::size_t* __restrict__ src_strides,
                                     std::size_t src_offset,
                                     int rank,
                                     std::size_t total) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total; i += stride) {
        // Convert flat index i → N-D multi-index → strided source offset.
        std::size_t src_idx = src_offset;
        std::size_t remaining = i;
        for (int d = rank - 1; d >= 0; --d) {
            const std::size_t dim_idx = remaining % shape[d];
            remaining /= shape[d];
            src_idx += dim_idx * src_strides[d];
        }
        dst[i] = src[src_idx];
    }
}
} // anon

// ---------------------------------------------------------------------------
//  Stride helpers
// ---------------------------------------------------------------------------
std::vector<std::size_t> Tensor::compute_contiguous_strides(
        const std::vector<std::size_t>& shape) {
    std::vector<std::size_t> strides(shape.size());
    if (!shape.empty()) {
        strides.back() = 1;
        for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i)
            strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

bool Tensor::is_contiguous() const noexcept {
    if (shape_.empty()) return true;
    auto ref = compute_contiguous_strides(shape_);
    return strides_ == ref && storage_offset_ == 0;
}

// ---------------------------------------------------------------------------
//  Constructors
// ---------------------------------------------------------------------------
Tensor::Tensor(std::vector<std::size_t> shape)
    : Tensor(std::move(shape), MemoryMode::host_and_device) {}

Tensor::Tensor(std::vector<std::size_t> shape, MemoryMode mode)
    : shape_(std::move(shape)), device_data_(nullptr, release_tensor_device),
      device_only_(mode == MemoryMode::device_only) {
    size_ = shape_.empty() ? 0 : std::accumulate(shape_.begin(), shape_.end(), std::size_t{1}, std::multiplies<>());
    strides_ = compute_contiguous_strides(shape_);
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
    : shape_(other.shape_), strides_(other.strides_), size_(other.size_),
      storage_offset_(0), host_data_(other.host_data_),
      device_data_(nullptr, release_tensor_device), device_only_(other.device_only_),
      host_current_(other.host_current_), is_view_(false) {
    // Always produce an owning contiguous copy.
    if (other.is_view_ || !other.is_contiguous()) {
        strides_ = compute_contiguous_strides(shape_);
        allocate_device();
        if (size_ != 0) {
            // Launch strided copy kernel.
            Tensor src_contig = other.contiguous();
            checkCuda(cudaMemcpyAsync(device_data_.get(), src_contig.device_data_.get(),
                                       size_ * sizeof(float), cudaMemcpyDeviceToDevice,
                                       compute_stream()), "Tensor copy contiguous");
        }
    } else {
        allocate_device();
        if (size_ != 0)
            checkCuda(cudaMemcpyAsync(device_data_.get(), other.device_data_.get(),
                                       size_ * sizeof(float), cudaMemcpyDeviceToDevice,
                                       compute_stream()), "Tensor device copy");
    }
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) *this = Tensor(other);
    return *this;
}

Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)), strides_(std::move(other.strides_)),
      size_(other.size_), storage_offset_(other.storage_offset_),
      host_data_(std::move(other.host_data_)),
      device_data_(std::move(other.device_data_)), device_only_(other.device_only_),
      host_current_(other.host_current_), is_view_(other.is_view_) {
    other.size_ = 0;
    other.storage_offset_ = 0;
    other.device_only_ = false;
    other.host_current_ = true;
    other.is_view_ = false;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        shape_ = std::move(other.shape_);
        strides_ = std::move(other.strides_);
        size_ = other.size_;
        storage_offset_ = other.storage_offset_;
        host_data_ = std::move(other.host_data_);
        device_data_ = std::move(other.device_data_);
        device_only_ = other.device_only_;
        host_current_ = other.host_current_;
        is_view_ = other.is_view_;
        other.size_ = 0;
        other.storage_offset_ = 0;
        other.device_only_ = false;
        other.host_current_ = true;
        other.is_view_ = false;
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
        // If non-contiguous, make contiguous first.
        if (!is_contiguous()) {
            *this = contiguous();
        }
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

// ---------------------------------------------------------------------------
//  contiguous() — copy non-contiguous tensor into contiguous storage
// ---------------------------------------------------------------------------
Tensor Tensor::contiguous() const {
    if (is_contiguous() && !is_view_) return *this;

    Tensor result(shape_, MemoryMode::device_only);
    if (size_ == 0) return result;

    // Upload shape and strides to device for the strided copy kernel.
    const int r = static_cast<int>(rank());
    const std::size_t meta_bytes = r * sizeof(std::size_t);
    std::size_t* d_shape = static_cast<std::size_t*>(allocate_device_memory(meta_bytes));
    std::size_t* d_strides = static_cast<std::size_t*>(allocate_device_memory(meta_bytes));
    checkCuda(cudaMemcpyAsync(d_shape, shape_.data(), meta_bytes,
                               cudaMemcpyHostToDevice, compute_stream()), "contiguous shape upload");
    checkCuda(cudaMemcpyAsync(d_strides, strides_.data(), meta_bytes,
                               cudaMemcpyHostToDevice, compute_stream()), "contiguous strides upload");

    const unsigned block = 256;
    const unsigned grid = static_cast<unsigned>(std::min(size_t((size_ + block - 1) / block),
                                                         size_t(sm_count() * 8)));
    strided_copy_kernel<<<grid, block, 0, compute_stream()>>>(
        device_data_.get(), result.device_data(), d_shape, d_strides, storage_offset_, r, size_);
    checkCuda(cudaGetLastError(), "strided_copy_kernel launch");

    free_device_memory(d_shape);
    free_device_memory(d_strides);

    result.mark_host_stale();
    return result;
}

// ---------------------------------------------------------------------------
//  permute — zero-copy dimension reordering
// ---------------------------------------------------------------------------
Tensor Tensor::permute(const std::vector<std::size_t>& dims) const {
    if (dims.size() != rank())
        throw InvalidArgumentError("permute: dims count must equal tensor rank");

    // Validate permutation.
    std::vector<bool> seen(rank(), false);
    for (auto d : dims) {
        if (d >= rank()) throw OutOfRangeError("permute: dimension index out of range");
        if (seen[d]) throw InvalidArgumentError("permute: duplicate dimension index");
        seen[d] = true;
    }

    Tensor result;
    result.shape_.resize(rank());
    result.strides_.resize(rank());
    for (std::size_t i = 0; i < rank(); ++i) {
        result.shape_[i] = shape_[dims[i]];
        result.strides_[i] = strides_[dims[i]];
    }
    result.size_ = size_;
    result.storage_offset_ = storage_offset_;
    result.device_only_ = true;
    result.host_current_ = false;
    result.is_view_ = true;
    // Share device storage (non-owning pointer).
    result.device_data_ = std::unique_ptr<float, void(*)(float*)>(
        device_data_.get(), noop_deleter);
    return result;
}

// ---------------------------------------------------------------------------
//  expand — zero-copy broadcast along size-1 dims
// ---------------------------------------------------------------------------
Tensor Tensor::expand(const std::vector<std::size_t>& new_shape) const {
    if (new_shape.size() != rank())
        throw InvalidArgumentError("expand: new_shape rank must match tensor rank");

    Tensor result;
    result.shape_ = new_shape;
    result.strides_.resize(rank());
    std::size_t new_size = 1;
    for (std::size_t i = 0; i < rank(); ++i) {
        if (shape_[i] == new_shape[i]) {
            result.strides_[i] = strides_[i];
        } else if (shape_[i] == 1) {
            result.strides_[i] = 0;  // broadcast — stride=0 repeats the data
        } else {
            throw ShapeMismatchError("expand: can only expand size-1 dimensions");
        }
        new_size *= new_shape[i];
    }
    result.size_ = new_size;
    result.storage_offset_ = storage_offset_;
    result.device_only_ = true;
    result.host_current_ = false;
    result.is_view_ = true;
    result.device_data_ = std::unique_ptr<float, void(*)(float*)>(
        device_data_.get(), noop_deleter);
    return result;
}

// ---------------------------------------------------------------------------
//  view — reshape contiguous tensor
// ---------------------------------------------------------------------------
Tensor Tensor::view(const std::vector<std::size_t>& new_shape) const {
    if (!is_contiguous())
        throw InvalidArgumentError("view: tensor must be contiguous; call contiguous() first");

    // Allow one -1 dimension (inferred).
    std::size_t new_size = 1;
    int infer_dim = -1;
    for (std::size_t i = 0; i < new_shape.size(); ++i) {
        if (new_shape[i] == static_cast<std::size_t>(-1)) {
            if (infer_dim >= 0)
                throw InvalidArgumentError("view: only one dimension can be inferred (-1)");
            infer_dim = static_cast<int>(i);
        } else {
            new_size *= new_shape[i];
        }
    }

    std::vector<std::size_t> resolved = new_shape;
    if (infer_dim >= 0) {
        if (new_size == 0 || size_ % new_size != 0)
            throw ShapeMismatchError("view: cannot infer dimension with given shape");
        resolved[infer_dim] = size_ / new_size;
    } else {
        if (new_size != size_)
            throw ShapeMismatchError("view: new shape element count must match tensor size");
    }

    Tensor result;
    result.shape_ = resolved;
    result.strides_ = compute_contiguous_strides(resolved);
    result.size_ = size_;
    result.storage_offset_ = storage_offset_;
    result.device_only_ = true;
    result.host_current_ = false;
    result.is_view_ = true;
    result.device_data_ = std::unique_ptr<float, void(*)(float*)>(
        device_data_.get(), noop_deleter);
    return result;
}

// ---------------------------------------------------------------------------
//  squeeze / unsqueeze
// ---------------------------------------------------------------------------
Tensor Tensor::squeeze(std::size_t dim) const {
    if (dim >= rank())
        throw OutOfRangeError("squeeze: dimension out of range");
    if (shape_[dim] != 1)
        return *this;  // no-op if dim is not size-1

    Tensor result;
    result.size_ = size_;
    result.storage_offset_ = storage_offset_;
    result.device_only_ = true;
    result.host_current_ = false;
    result.is_view_ = true;
    result.device_data_ = std::unique_ptr<float, void(*)(float*)>(
        device_data_.get(), noop_deleter);
    for (std::size_t i = 0; i < rank(); ++i) {
        if (i != dim) {
            result.shape_.push_back(shape_[i]);
            result.strides_.push_back(strides_[i]);
        }
    }
    return result;
}

Tensor Tensor::unsqueeze(std::size_t dim) const {
    if (dim > rank())
        throw OutOfRangeError("unsqueeze: dimension out of range");

    Tensor result;
    result.size_ = size_;
    result.storage_offset_ = storage_offset_;
    result.device_only_ = true;
    result.host_current_ = false;
    result.is_view_ = true;
    result.device_data_ = std::unique_ptr<float, void(*)(float*)>(
        device_data_.get(), noop_deleter);
    result.shape_ = shape_;
    result.strides_ = strides_;
    // Insert size-1 dimension with stride = next dimension's stride * size (or 1 if last).
    std::size_t new_stride = (dim < rank()) ? strides_[dim] * shape_[dim] : 1;
    result.shape_.insert(result.shape_.begin() + dim, 1);
    result.strides_.insert(result.strides_.begin() + dim, new_stride);
    return result;
}

// ---------------------------------------------------------------------------
//  narrow — zero-copy slice along one dimension
// ---------------------------------------------------------------------------
Tensor Tensor::narrow(std::size_t dim, std::size_t start, std::size_t length) const {
    if (dim >= rank())
        throw OutOfRangeError("narrow: dimension out of range");
    if (start + length > shape_[dim])
        throw OutOfRangeError("narrow: start + length exceeds dimension size");

    Tensor result;
    result.shape_ = shape_;
    result.strides_ = strides_;
    result.shape_[dim] = length;
    result.storage_offset_ = storage_offset_ + start * strides_[dim];
    result.device_only_ = true;
    result.host_current_ = false;
    result.is_view_ = true;
    result.device_data_ = std::unique_ptr<float, void(*)(float*)>(
        device_data_.get(), noop_deleter);
    result.size_ = 1;
    for (auto s : result.shape_) result.size_ *= s;
    return result;
}

// ---------------------------------------------------------------------------
//  select — pick one index along a dimension, reducing rank by 1
// ---------------------------------------------------------------------------
Tensor Tensor::select(std::size_t dim, std::size_t index) const {
    if (dim >= rank())
        throw OutOfRangeError("select: dimension out of range");
    if (index >= shape_[dim])
        throw OutOfRangeError("select: index out of range for dimension");

    Tensor result;
    result.storage_offset_ = storage_offset_ + index * strides_[dim];
    result.device_only_ = true;
    result.host_current_ = false;
    result.is_view_ = true;
    result.device_data_ = std::unique_ptr<float, void(*)(float*)>(
        device_data_.get(), noop_deleter);

    for (std::size_t i = 0; i < rank(); ++i) {
        if (i != dim) {
            result.shape_.push_back(shape_[i]);
            result.strides_.push_back(strides_[i]);
        }
    }
    result.size_ = result.shape_.empty() ? 1 :
        std::accumulate(result.shape_.begin(), result.shape_.end(), std::size_t{1}, std::multiplies<>());
    return result;
}

}