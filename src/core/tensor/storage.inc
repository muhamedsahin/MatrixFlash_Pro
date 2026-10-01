// Dahili uygulama parçası: src/core/tensor.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Stride hesapları ve Tensor kurucu/kopya/taşıma sahipliği.

// ---------------------------------------------------------------------------
//  Stride helpers
// ---------------------------------------------------------------------------
std::vector<std::size_t> Tensor::compute_contiguous_strides(const std::vector<std::size_t>& shape) {
    std::vector<std::size_t> strides(shape.size());
    if (!shape.empty()) {
        strides.back() = 1;
        for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i)
            strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

bool Tensor::is_contiguous() const noexcept {
    if (shape_.empty())
        return true;
    auto ref = compute_contiguous_strides(shape_);
    return strides_ == ref && storage_offset_ == 0;
}

// ---------------------------------------------------------------------------
//  Constructors
// ---------------------------------------------------------------------------
Tensor::Tensor(std::vector<std::size_t> shape)
    : Tensor(std::move(shape), MemoryMode::host_and_device) {
}

Tensor::Tensor(std::vector<std::size_t> shape, MemoryMode mode)
    : shape_(std::move(shape)), device_data_(nullptr, release_tensor_device),
      device_only_(mode == MemoryMode::device_only) {
    size_ = shape_.empty() ? 0
                           : std::accumulate(shape_.begin(), shape_.end(), std::size_t{1},
                                             std::multiplies<>());
    strides_ = compute_contiguous_strides(shape_);
    if (!device_only_)
        host_data_.resize(size_);
    allocate_device();
}

Tensor::Tensor(std::vector<std::size_t> shape, const std::vector<float>& values)
    : Tensor(std::move(shape)) {
    if (values.size() != size_)
        throw ShapeMismatchError("Tensor data size does not match shape");
    host_data_ = values;
    upload();
}

Tensor::Tensor(const Tensor& other)
    : shape_(other.shape_), strides_(other.strides_), size_(other.size_), storage_offset_(0),
      host_data_(other.host_data_), device_data_(nullptr, release_tensor_device),
      device_only_(other.device_only_), host_current_(other.host_current_), is_view_(false) {
    // Always produce an owning contiguous copy.
    if (other.is_view_ || !other.is_contiguous()) {
        strides_ = compute_contiguous_strides(shape_);
        allocate_device();
        if (size_ != 0) {
            // Launch strided copy kernel.
            Tensor src_contig = other.contiguous();
            checkCuda(cudaMemcpyAsync(device_data_.get(), src_contig.device_data_.get(),
                                      size_ * sizeof(float), cudaMemcpyDeviceToDevice,
                                      compute_stream()),
                      "Tensor copy contiguous");
        }
    } else {
        allocate_device();
        if (size_ != 0)
            checkCuda(cudaMemcpyAsync(device_data_.get(), other.device_data_.get(),
                                      size_ * sizeof(float), cudaMemcpyDeviceToDevice,
                                      compute_stream()),
                      "Tensor device copy");
    }
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other)
        *this = Tensor(other);
    return *this;
}

Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)), strides_(std::move(other.strides_)), size_(other.size_),
      storage_offset_(other.storage_offset_), host_data_(std::move(other.host_data_)),
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
