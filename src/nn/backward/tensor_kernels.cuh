// Dahili uygulama parçası: src/nn/conv_backward.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Tensor elementwise, skaler ve transpose launch yardımcıları.

// --- tensor elementwise kernels ---
__global__ void tensor_binary_kernel(const float* left, const float* right, float* output,
                                     std::size_t count, bool add) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count)
        return;
    output[index] = add ? left[index] + right[index] : left[index] * right[index];
}

__global__ void tensor_scalar_kernel(const float* input, float* output, std::size_t count,
                                     float scalar) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count)
        return;
    output[index] = input[index] * scalar;
}

void validate_pool(const Tensor& grad, const Tensor& input, std::size_t kernel_size,
                   std::size_t stride, std::size_t padding, std::size_t& out_h,
                   std::size_t& out_w) {
    if (input.rank() != 4 || grad.rank() != 4)
        throw InvalidArgumentError("pool backward expects rank-4 tensors");
    if (kernel_size == 0 || stride == 0)
        throw InvalidArgumentError("pool backward kernel/stride must be positive");
    const auto& x = input.shape();
    const auto& g = grad.shape();
    out_h = (x[2] + 2 * padding - kernel_size) / stride + 1;
    out_w = (x[3] + 2 * padding - kernel_size) / stride + 1;
    if (g[0] != x[0] || g[1] != x[1] || g[2] != out_h || g[3] != out_w)
        throw ShapeMismatchError("pool backward gradient shape mismatch");
}

Tensor tensor_elementwise(const Tensor& left, const Tensor& right, bool add, const char* message) {
    if (left.shape() != right.shape())
        throw InvalidArgumentError(message);
    Tensor output(left.shape());
    tensor_binary_kernel<<<static_cast<unsigned>((left.size() + 255) / 256), 256, 0,
                           compute_stream()>>>(left.device_data(), right.device_data(),
                                               output.device_data(), left.size(), add);
    checkCuda(cudaGetLastError(), "tensor elementwise kernel launch");
    output.mark_host_stale();
    return output;
}
