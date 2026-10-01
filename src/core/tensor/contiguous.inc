// Dahili uygulama parçası: src/core/tensor.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Noncontiguous görünümu cihazda bitişik depolamaya kopyalar.

// ---------------------------------------------------------------------------
//  contiguous() — copy non-contiguous tensor into contiguous storage
// ---------------------------------------------------------------------------
Tensor Tensor::contiguous() const {
    if (is_contiguous() && !is_view_)
        return *this;

    Tensor result(shape_, MemoryMode::device_only);
    if (size_ == 0)
        return result;

    // Upload shape and strides to device for the strided copy kernel.
    const int r = static_cast<int>(rank());
    const std::size_t meta_bytes = r * sizeof(std::size_t);
    std::size_t* d_shape = static_cast<std::size_t*>(allocate_device_memory(meta_bytes));
    std::size_t* d_strides = static_cast<std::size_t*>(allocate_device_memory(meta_bytes));
    checkCuda(cudaMemcpyAsync(d_shape, shape_.data(), meta_bytes, cudaMemcpyHostToDevice,
                              compute_stream()),
              "contiguous shape upload");
    checkCuda(cudaMemcpyAsync(d_strides, strides_.data(), meta_bytes, cudaMemcpyHostToDevice,
                              compute_stream()),
              "contiguous strides upload");

    const unsigned block = 256;
    const unsigned grid = static_cast<unsigned>(
        std::min(size_t((size_ + block - 1) / block), size_t(sm_count() * 8)));
    strided_copy_kernel<<<grid, block, 0, compute_stream()>>>(
        device_data_.get(), result.device_data(), d_shape, d_strides, storage_offset_, r, size_);
    checkCuda(cudaGetLastError(), "strided_copy_kernel launch");

    free_device_memory(d_shape);
    free_device_memory(d_strides);

    result.mark_host_stale();
    return result;
}
