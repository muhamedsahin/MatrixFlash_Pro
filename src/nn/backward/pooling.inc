// Dahili uygulama parçası: src/nn/conv_backward.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Max ve average pool backward erişimleri.

Tensor max_pool2d_backward(const Tensor& grad, const Tensor& input, std::size_t kernel_size,
                           std::size_t stride, std::size_t padding) {
    std::size_t out_h = 0, out_w = 0;
    validate_pool(grad, input, kernel_size, stride, padding, out_h, out_w);
    const auto& x = input.shape();
    Tensor input_grad(x);
    // The kernel scatters (atomicAdd) into the buffer, so it must start at zero.
    zero_device_memory(input_grad.device_data(), input_grad.size() * sizeof(float));
    max_pool2d_backward_kernel<<<static_cast<unsigned>((grad.size() + 255) / 256), 256, 0,
                                 compute_stream()>>>(
        grad.device_data(), input.device_data(), input_grad.device_data(), x[0], x[1], x[2], x[3],
        out_h, out_w, kernel_size, stride, padding);
    checkCuda(cudaGetLastError(), "max pool backward kernel launch");
    input_grad.mark_host_stale();
    return input_grad;
}

Tensor avg_pool2d_backward(const Tensor& grad, const Tensor& input, std::size_t kernel_size,
                           std::size_t stride, std::size_t padding) {
    std::size_t out_h = 0, out_w = 0;
    validate_pool(grad, input, kernel_size, stride, padding, out_h, out_w);
    const auto& x = input.shape();
    Tensor input_grad(x);
    // The kernel scatters (atomicAdd) into the buffer, so it must start at zero.
    zero_device_memory(input_grad.device_data(), input_grad.size() * sizeof(float));
    avg_pool2d_backward_kernel<<<static_cast<unsigned>((grad.size() + 255) / 256), 256, 0,
                                 compute_stream()>>>(grad.device_data(), input_grad.device_data(),
                                                     x[0], x[1], x[2], x[3], out_h, out_w,
                                                     kernel_size, stride, padding);
    checkCuda(cudaGetLastError(), "avg pool backward kernel launch");
    input_grad.mark_host_stale();
    return input_grad;
}
