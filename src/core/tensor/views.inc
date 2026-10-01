// Dahili uygulama parçası: src/core/tensor.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Permute, expand, reshape ve boyut görünümleri; cihaz verisi kopyalanmaz.

// ---------------------------------------------------------------------------
//  permute — zero-copy dimension reordering
// ---------------------------------------------------------------------------
Tensor Tensor::permute(const std::vector<std::size_t>& dims) const {
    if (dims.size() != rank())
        throw InvalidArgumentError("permute: dims count must equal tensor rank");

    // Validate permutation.
    std::vector<bool> seen(rank(), false);
    for (auto d : dims) {
        if (d >= rank())
            throw OutOfRangeError("permute: dimension index out of range");
        if (seen[d])
            throw InvalidArgumentError("permute: duplicate dimension index");
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
    result.device_data_ =
        std::unique_ptr<float, void (*)(float*)>(device_data_.get(), noop_deleter);
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
            result.strides_[i] = 0; // broadcast — stride=0 repeats the data
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
    result.device_data_ =
        std::unique_ptr<float, void (*)(float*)>(device_data_.get(), noop_deleter);
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
    result.device_data_ =
        std::unique_ptr<float, void (*)(float*)>(device_data_.get(), noop_deleter);
    return result;
}

// ---------------------------------------------------------------------------
//  squeeze / unsqueeze
// ---------------------------------------------------------------------------
Tensor Tensor::squeeze(std::size_t dim) const {
    if (dim >= rank())
        throw OutOfRangeError("squeeze: dimension out of range");
    if (shape_[dim] != 1)
        return *this; // no-op if dim is not size-1

    Tensor result;
    result.size_ = size_;
    result.storage_offset_ = storage_offset_;
    result.device_only_ = true;
    result.host_current_ = false;
    result.is_view_ = true;
    result.device_data_ =
        std::unique_ptr<float, void (*)(float*)>(device_data_.get(), noop_deleter);
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
    result.device_data_ =
        std::unique_ptr<float, void (*)(float*)>(device_data_.get(), noop_deleter);
    result.shape_ = shape_;
    result.strides_ = strides_;
    // Insert size-1 dimension with stride = next dimension's stride * size (or 1 if last).
    std::size_t new_stride = (dim < rank()) ? strides_[dim] * shape_[dim] : 1;
    result.shape_.insert(result.shape_.begin() + dim, 1);
    result.strides_.insert(result.strides_.begin() + dim, new_stride);
    return result;
}
