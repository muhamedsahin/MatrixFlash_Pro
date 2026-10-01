// Dahili uygulama parçası: src/nn/conv_backward.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Conv input/weight/bias backward fonksiyonları.

Tensor conv2d_input_backward(const Tensor& grad, const Tensor& input, const Tensor& weights,
                             std::size_t stride, std::size_t padding) {
    std::size_t out_h = 0, out_w = 0;
    validate_conv(grad, input, weights, out_h, out_w, stride, padding);
    const auto& x = input.shape();
    const auto& w = weights.shape();
    Tensor input_grad(x);
    conv2d_input_backward_kernel<<<static_cast<unsigned>((input_grad.size() + 255) / 256), 256, 0,
                                   compute_stream()>>>(
        grad.device_data(), weights.device_data(), input_grad.device_data(), x[0], x[1], x[2], x[3],
        w[0], w[2], w[3], out_h, out_w, stride, padding);
    checkCuda(cudaGetLastError(), "conv2d input backward kernel launch");
    input_grad.mark_host_stale();
    return input_grad;
}

Tensor conv2d_weight_backward(const Tensor& grad, const Tensor& input, const Tensor& weights,
                              std::size_t stride, std::size_t padding) {
    std::size_t out_h = 0, out_w = 0;
    validate_conv(grad, input, weights, out_h, out_w, stride, padding);
    const auto& x = input.shape();
    const auto& w = weights.shape();
    Tensor weight_grad(w);
    conv2d_weight_backward_kernel<<<static_cast<unsigned>((weight_grad.size() + 255) / 256), 256, 0,
                                    compute_stream()>>>(
        grad.device_data(), input.device_data(), weight_grad.device_data(), x[0], x[1], x[2], x[3],
        w[0], w[2], w[3], out_h, out_w, stride, padding);
    checkCuda(cudaGetLastError(), "conv2d weight backward kernel launch");
    weight_grad.mark_host_stale();
    return weight_grad;
}

Tensor conv2d_bias_backward(const Tensor& grad) {
    if (grad.rank() != 4)
        throw InvalidArgumentError("conv2d bias backward expects a rank-4 gradient");
    const auto& g = grad.shape();
    Tensor bias_grad({g[1]});
    conv2d_bias_backward_kernel<<<static_cast<unsigned>((g[1] + 255) / 256), 256, 0,
                                  compute_stream()>>>(grad.device_data(), bias_grad.device_data(),
                                                      g[0], g[1], g[2], g[3]);
    checkCuda(cudaGetLastError(), "conv2d bias backward kernel launch");
    bias_grad.mark_host_stale();
    return bias_grad;
}
