// Dahili uygulama parçası: src/nn/conv_backward.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Tensor aritmetik primitifleri; autograd bunları kullanır.

Tensor tensor_add(const Tensor& left, const Tensor& right) {
    return tensor_elementwise(left, right, true, "tensor_add shape mismatch");
}

Tensor tensor_multiply(const Tensor& left, const Tensor& right) {
    return tensor_elementwise(left, right, false, "tensor_multiply shape mismatch");
}

Tensor tensor_multiply(const Tensor& tensor, float scalar) {
    Tensor output(tensor.shape());
    tensor_scalar_kernel<<<static_cast<unsigned>((tensor.size() + 255) / 256), 256, 0,
                           compute_stream()>>>(tensor.device_data(), output.device_data(),
                                               tensor.size(), scalar);
    checkCuda(cudaGetLastError(), "tensor scalar kernel launch");
    output.mark_host_stale();
    return output;
}

Tensor tensor_negate(const Tensor& tensor) {
    return tensor_multiply(tensor, -1.0f);
}
