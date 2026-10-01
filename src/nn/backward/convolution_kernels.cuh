// Dahili uygulama parçası: src/nn/conv_backward.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: NCHW input, weight ve bias gradyan çekirdekleri ve şekil denetimi.

namespace {

// dL/dinput[n,c,ih,iw] = sum over filters & valid window positions of
// grad[n,f,oh,ow] * weights[f,c,kh,kw].
__global__ void conv2d_input_backward_kernel(const float* grad, const float* weights,
                                             float* input_grad, std::size_t batch,
                                             std::size_t channels, std::size_t height,
                                             std::size_t width, std::size_t filters,
                                             std::size_t kh_size, std::size_t kw_size,
                                             std::size_t out_h, std::size_t out_w,
                                             std::size_t stride, std::size_t padding) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto total = batch * channels * height * width;
    if (index >= total)
        return;
    const std::size_t iw = index % width;
    const std::size_t ih = (index / width) % height;
    const std::size_t channel = (index / (width * height)) % channels;
    const std::size_t sample = index / (channels * width * height);
    float accumulator = 0.0f;
    for (std::size_t filter = 0; filter < filters; ++filter) {
        for (std::size_t kh = 0; kh < kh_size; ++kh) {
            const long oh_long =
                static_cast<long>(ih) + static_cast<long>(padding) - static_cast<long>(kh);
            if (oh_long < 0 || oh_long % static_cast<long>(stride) != 0)
                continue;
            const std::size_t oh = static_cast<std::size_t>(oh_long) / stride;
            if (oh >= out_h)
                continue;
            for (std::size_t kw = 0; kw < kw_size; ++kw) {
                const long ow_long =
                    static_cast<long>(iw) + static_cast<long>(padding) - static_cast<long>(kw);
                if (ow_long < 0 || ow_long % static_cast<long>(stride) != 0)
                    continue;
                const std::size_t ow = static_cast<std::size_t>(ow_long) / stride;
                if (ow >= out_w)
                    continue;
                const auto grad_index = ((sample * filters + filter) * out_h + oh) * out_w + ow;
                const auto weight_index =
                    ((filter * channels + channel) * kh_size + kh) * kw_size + kw;
                accumulator += grad[grad_index] * weights[weight_index];
            }
        }
    }
    input_grad[index] = accumulator;
}

// dL/dweight[f,c,kh,kw] = sum over batch & positions of grad[n,f,oh,ow] * input[n,c,ih,iw].
__global__ void conv2d_weight_backward_kernel(const float* grad, const float* input,
                                              float* weight_grad, std::size_t batch,
                                              std::size_t channels, std::size_t height,
                                              std::size_t width, std::size_t filters,
                                              std::size_t kh_size, std::size_t kw_size,
                                              std::size_t out_h, std::size_t out_w,
                                              std::size_t stride, std::size_t padding) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto total = filters * channels * kh_size * kw_size;
    if (index >= total)
        return;
    const std::size_t kw = index % kw_size;
    const std::size_t kh = (index / kw_size) % kh_size;
    const std::size_t channel = (index / (kw_size * kh_size)) % channels;
    const std::size_t filter = index / (channels * kh_size * kw_size);
    float accumulator = 0.0f;
    for (std::size_t sample = 0; sample < batch; ++sample) {
        for (std::size_t oh = 0; oh < out_h; ++oh) {
            const long ih_long = static_cast<long>(oh * stride + kh) - static_cast<long>(padding);
            if (ih_long < 0 || ih_long >= static_cast<long>(height))
                continue;
            for (std::size_t ow = 0; ow < out_w; ++ow) {
                const long iw_long =
                    static_cast<long>(ow * stride + kw) - static_cast<long>(padding);
                if (iw_long < 0 || iw_long >= static_cast<long>(width))
                    continue;
                const auto grad_index = ((sample * filters + filter) * out_h + oh) * out_w + ow;
                const auto input_index =
                    ((sample * channels + channel) * height + static_cast<std::size_t>(ih_long)) *
                        width +
                    static_cast<std::size_t>(iw_long);
                accumulator += grad[grad_index] * input[input_index];
            }
        }
    }
    weight_grad[index] = accumulator;
}

// dL/dbias[f] = sum over batch & positions of grad[n,f,oh,ow].
__global__ void conv2d_bias_backward_kernel(const float* grad, float* bias_grad, std::size_t batch,
                                            std::size_t filters, std::size_t out_h,
                                            std::size_t out_w) {
    const auto filter = blockIdx.x * blockDim.x + threadIdx.x;
    if (filter >= filters)
        return;
    float accumulator = 0.0f;
    for (std::size_t sample = 0; sample < batch; ++sample)
        for (std::size_t oh = 0; oh < out_h; ++oh)
            for (std::size_t ow = 0; ow < out_w; ++ow)
                accumulator += grad[(sample * filters + filter) * out_h * out_w + oh * out_w + ow];
    bias_grad[filter] = accumulator;
}

void validate_conv(const Tensor& grad, const Tensor& input, const Tensor& weights,
                   std::size_t& out_h, std::size_t& out_w, std::size_t stride,
                   std::size_t padding) {
    if (input.rank() != 4)
        throw InvalidArgumentError("conv2d backward input must be [N,C,H,W]");
    if (weights.rank() != 4 || grad.rank() != 4)
        throw InvalidArgumentError("conv2d backward expects rank-4 grad/weights");
    const auto& x = input.shape();
    const auto& w = weights.shape();
    const auto& g = grad.shape();
    if (w[1] != x[1] || g[0] != x[0] || g[1] != w[0])
        throw ShapeMismatchError("conv2d backward shape mismatch");
    out_h = (x[2] + 2 * padding - w[2]) / stride + 1;
    out_w = (x[3] + 2 * padding - w[3]) / stride + 1;
    if (g[2] != out_h || g[3] != out_w)
        throw ShapeMismatchError("conv2d backward grad spatial shape mismatch");
}

} // anonymous namespace (conv kernels)
