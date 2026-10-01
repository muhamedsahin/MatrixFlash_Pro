// Dahili uygulama parçası: src/core/tensor.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Narrow ve select: stride/storage_offset ile sıfır kopya dilimleme.

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
    result.device_data_ =
        std::unique_ptr<float, void (*)(float*)>(device_data_.get(), noop_deleter);
    result.size_ = 1;
    for (auto s : result.shape_)
        result.size_ *= s;
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
    result.device_data_ =
        std::unique_ptr<float, void (*)(float*)>(device_data_.get(), noop_deleter);

    for (std::size_t i = 0; i < rank(); ++i) {
        if (i != dim) {
            result.shape_.push_back(shape_[i]);
            result.strides_.push_back(strides_[i]);
        }
    }
    result.size_ = result.shape_.empty()
                       ? 1
                       : std::accumulate(result.shape_.begin(), result.shape_.end(), std::size_t{1},
                                         std::multiplies<>());
    return result;
}
