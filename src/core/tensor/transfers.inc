// Dahili uygulama parçası: src/core/tensor.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Upload/download ve host aynası güncelliği; örtük indirme eklenmez.

void Tensor::allocate_device() {
    if (size_ != 0)
        device_data_.reset(static_cast<float*>(allocate_device_memory(size_ * sizeof(float))));
}

void Tensor::ensure_host_buffer() {
    if (host_data_.size() != size_)
        host_data_.resize(size_);
}

void Tensor::upload() {
    if (size_ != 0 && !device_only_) {
        checkCuda(cudaMemcpyAsync(device_data_.get(), host_data_.data(), size_ * sizeof(float),
                                  cudaMemcpyHostToDevice, compute_stream()),
                  "Tensor upload");
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
        checkCuda(cudaMemcpyAsync(host_data_.data(), device_data_.get(), size_ * sizeof(float),
                                  cudaMemcpyDeviceToHost, compute_stream()),
                  "Tensor download");
        checkCuda(cudaStreamSynchronize(compute_stream()), "Tensor download synchronize");
        mark_host_current();
    }
}

const std::vector<float>& Tensor::data() const {
    if (host_data_.size() != size_)
        throw InvalidArgumentError("Tensor::data() has no host data; call download() first");
    if (!host_current_)
        throw InvalidArgumentError(
            "Tensor::data() host mirror is stale (device holds newer data); call download() first");
    return host_data_;
}

void Tensor::fill(float value) {
    if (size_ == 0)
        return;
    if (host_materialized()) {
        std::fill(host_data_.begin(), host_data_.end(), value);
        upload();
    } else {
        // device_only with no host mirror yet: write straight to the device.
        tensor_fill_kernel<<<static_cast<unsigned>((size_ + 255) / 256), 256, 0,
                             compute_stream()>>>(device_data_.get(), size_, value);
        checkCuda(cudaGetLastError(), "Tensor fill kernel launch");
    }
}

void Tensor::zero() {
    if (!host_data_.empty())
        std::fill(host_data_.begin(), host_data_.end(), 0.0f);
    if (size_ != 0)
        zero_device_memory(device_data_.get(), size_ * sizeof(float));
    // Both mirrors now hold zeros: the mirror is current again.
    if (host_materialized())
        mark_host_current();
}

void Tensor::synchronize() const {
    matrix_pro::synchronize();
}
