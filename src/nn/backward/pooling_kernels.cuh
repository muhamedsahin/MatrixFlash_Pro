// Dahili uygulama parçası: src/nn/conv_backward.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Overlap için atomicAdd; max argmax ve padding semantiği korunur.

// Max pooling backward: recompute the argmax inside each window and scatter
// the upstream gradient there (windows may overlap -> atomicAdd).
__global__ void max_pool2d_backward_kernel(const float* grad, const float* input, float* input_grad,
                                           std::size_t batch, std::size_t channels,
                                           std::size_t height, std::size_t width, std::size_t out_h,
                                           std::size_t out_w, std::size_t kernel_size,
                                           std::size_t stride, std::size_t padding) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto total = batch * channels * out_h * out_w;
    if (index >= total)
        return;
    const std::size_t ow = index % out_w;
    const std::size_t oh = (index / out_w) % out_h;
    const std::size_t channel = (index / (out_w * out_h)) % channels;
    const std::size_t sample = index / (channels * out_w * out_h);
    float best = -FLT_MAX;
    std::size_t best_index = 0;
    bool found = false;
    for (std::size_t kh = 0; kh < kernel_size; ++kh) {
        const long input_row = static_cast<long>(oh * stride + kh) - static_cast<long>(padding);
        if (input_row < 0 || input_row >= static_cast<long>(height))
            continue;
        for (std::size_t kw = 0; kw < kernel_size; ++kw) {
            const long input_col = static_cast<long>(ow * stride + kw) - static_cast<long>(padding);
            if (input_col < 0 || input_col >= static_cast<long>(width))
                continue;
            const auto input_index =
                ((sample * channels + channel) * height + static_cast<std::size_t>(input_row)) *
                    width +
                static_cast<std::size_t>(input_col);
            const float value = input[input_index];
            if (!found || value > best) {
                best = value;
                best_index = input_index;
                found = true;
            }
        }
    }
    if (found)
        atomicAdd(input_grad + best_index, grad[index]);
}

// Average pooling backward (count-include-padding semantics, matches forward).
__global__ void avg_pool2d_backward_kernel(const float* grad, float* input_grad, std::size_t batch,
                                           std::size_t channels, std::size_t height,
                                           std::size_t width, std::size_t out_h, std::size_t out_w,
                                           std::size_t kernel_size, std::size_t stride,
                                           std::size_t padding) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto total = batch * channels * out_h * out_w;
    if (index >= total)
        return;
    const std::size_t ow = index % out_w;
    const std::size_t oh = (index / out_w) % out_h;
    const std::size_t channel = (index / (out_w * out_h)) % channels;
    const std::size_t sample = index / (channels * out_w * out_h);
    const float share = grad[index] / static_cast<float>(kernel_size * kernel_size);
    for (std::size_t kh = 0; kh < kernel_size; ++kh) {
        const long input_row = static_cast<long>(oh * stride + kh) - static_cast<long>(padding);
        if (input_row < 0 || input_row >= static_cast<long>(height))
            continue;
        for (std::size_t kw = 0; kw < kernel_size; ++kw) {
            const long input_col = static_cast<long>(ow * stride + kw) - static_cast<long>(padding);
            if (input_col < 0 || input_col >= static_cast<long>(width))
                continue;
            const auto input_index =
                ((sample * channels + channel) * height + static_cast<std::size_t>(input_row)) *
                    width +
                static_cast<std::size_t>(input_col);
            atomicAdd(input_grad + input_index, share);
        }
    }
}
