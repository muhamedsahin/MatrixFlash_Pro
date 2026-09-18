#include "matrix_pro/ops/operations.hpp"
#include "matrix_pro/core/cuda_utils.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cfloat>
#include <stdexcept>

namespace matrix_pro {
namespace {

// dL/dinput[n,c,ih,iw] = sum over filters & valid window positions of
// grad[n,f,oh,ow] * weights[f,c,kh,kw].
__global__ void conv2d_input_backward_kernel(const float* grad, const float* weights, float* input_grad,
                                             std::size_t batch, std::size_t channels, std::size_t height, std::size_t width,
                                             std::size_t filters, std::size_t kh_size, std::size_t kw_size,
                                             std::size_t out_h, std::size_t out_w,
                                             std::size_t stride, std::size_t padding) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto total = batch * channels * height * width;
    if (index >= total) return;
    const std::size_t iw = index % width;
    const std::size_t ih = (index / width) % height;
    const std::size_t channel = (index / (width * height)) % channels;
    const std::size_t sample = index / (channels * width * height);
    float accumulator = 0.0f;
    for (std::size_t filter = 0; filter < filters; ++filter) {
        for (std::size_t kh = 0; kh < kh_size; ++kh) {
            const long oh_long = static_cast<long>(ih) + static_cast<long>(padding) - static_cast<long>(kh);
            if (oh_long < 0 || oh_long % static_cast<long>(stride) != 0) continue;
            const std::size_t oh = static_cast<std::size_t>(oh_long) / stride;
            if (oh >= out_h) continue;
            for (std::size_t kw = 0; kw < kw_size; ++kw) {
                const long ow_long = static_cast<long>(iw) + static_cast<long>(padding) - static_cast<long>(kw);
                if (ow_long < 0 || ow_long % static_cast<long>(stride) != 0) continue;
                const std::size_t ow = static_cast<std::size_t>(ow_long) / stride;
                if (ow >= out_w) continue;
                const auto grad_index = ((sample * filters + filter) * out_h + oh) * out_w + ow;
                const auto weight_index = ((filter * channels + channel) * kh_size + kh) * kw_size + kw;
                accumulator += grad[grad_index] * weights[weight_index];
            }
        }
    }
    input_grad[index] = accumulator;
}

// dL/dweight[f,c,kh,kw] = sum over batch & positions of grad[n,f,oh,ow] * input[n,c,ih,iw].
__global__ void conv2d_weight_backward_kernel(const float* grad, const float* input, float* weight_grad,
                                              std::size_t batch, std::size_t channels, std::size_t height, std::size_t width,
                                              std::size_t filters, std::size_t kh_size, std::size_t kw_size,
                                              std::size_t out_h, std::size_t out_w,
                                              std::size_t stride, std::size_t padding) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto total = filters * channels * kh_size * kw_size;
    if (index >= total) return;
    const std::size_t kw = index % kw_size;
    const std::size_t kh = (index / kw_size) % kh_size;
    const std::size_t channel = (index / (kw_size * kh_size)) % channels;
    const std::size_t filter = index / (channels * kh_size * kw_size);
    float accumulator = 0.0f;
    for (std::size_t sample = 0; sample < batch; ++sample) {
        for (std::size_t oh = 0; oh < out_h; ++oh) {
            const long ih_long = static_cast<long>(oh * stride + kh) - static_cast<long>(padding);
            if (ih_long < 0 || ih_long >= static_cast<long>(height)) continue;
            for (std::size_t ow = 0; ow < out_w; ++ow) {
                const long iw_long = static_cast<long>(ow * stride + kw) - static_cast<long>(padding);
                if (iw_long < 0 || iw_long >= static_cast<long>(width)) continue;
                const auto grad_index = ((sample * filters + filter) * out_h + oh) * out_w + ow;
                const auto input_index = ((sample * channels + channel) * height + static_cast<std::size_t>(ih_long)) * width + static_cast<std::size_t>(iw_long);
                accumulator += grad[grad_index] * input[input_index];
            }
        }
    }
    weight_grad[index] = accumulator;
}

// dL/dbias[f] = sum over batch & positions of grad[n,f,oh,ow].
__global__ void conv2d_bias_backward_kernel(const float* grad, float* bias_grad,
                                            std::size_t batch, std::size_t filters,
                                            std::size_t out_h, std::size_t out_w) {
    const auto filter = blockIdx.x * blockDim.x + threadIdx.x;
    if (filter >= filters) return;
    float accumulator = 0.0f;
    for (std::size_t sample = 0; sample < batch; ++sample)
        for (std::size_t oh = 0; oh < out_h; ++oh)
            for (std::size_t ow = 0; ow < out_w; ++ow)
                accumulator += grad[(sample * filters + filter) * out_h * out_w + oh * out_w + ow];
    bias_grad[filter] = accumulator;
}

void validate_conv(const Tensor& grad, const Tensor& input, const Tensor& weights,
                   std::size_t& out_h, std::size_t& out_w, std::size_t stride, std::size_t padding) {
    if (input.rank() != 4) throw InvalidArgumentError("conv2d backward input must be [N,C,H,W]");
    if (weights.rank() != 4 || grad.rank() != 4) throw InvalidArgumentError("conv2d backward expects rank-4 grad/weights");
    const auto& x = input.shape();
    const auto& w = weights.shape();
    const auto& g = grad.shape();
    if (w[1] != x[1] || g[0] != x[0] || g[1] != w[0]) throw ShapeMismatchError("conv2d backward shape mismatch");
    out_h = (x[2] + 2 * padding - w[2]) / stride + 1;
    out_w = (x[3] + 2 * padding - w[3]) / stride + 1;
    if (g[2] != out_h || g[3] != out_w) throw ShapeMismatchError("conv2d backward grad spatial shape mismatch");
}

}  // anonymous namespace (conv kernels)

namespace {

// Max pooling backward: recompute the argmax inside each window and scatter
// the upstream gradient there (windows may overlap -> atomicAdd).
__global__ void max_pool2d_backward_kernel(const float* grad, const float* input, float* input_grad,
                                           std::size_t batch, std::size_t channels, std::size_t height, std::size_t width,
                                           std::size_t out_h, std::size_t out_w,
                                           std::size_t kernel_size, std::size_t stride, std::size_t padding) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto total = batch * channels * out_h * out_w;
    if (index >= total) return;
    const std::size_t ow = index % out_w;
    const std::size_t oh = (index / out_w) % out_h;
    const std::size_t channel = (index / (out_w * out_h)) % channels;
    const std::size_t sample = index / (channels * out_w * out_h);
    float best = -FLT_MAX;
    std::size_t best_index = 0;
    bool found = false;
    for (std::size_t kh = 0; kh < kernel_size; ++kh) {
        const long input_row = static_cast<long>(oh * stride + kh) - static_cast<long>(padding);
        if (input_row < 0 || input_row >= static_cast<long>(height)) continue;
        for (std::size_t kw = 0; kw < kernel_size; ++kw) {
            const long input_col = static_cast<long>(ow * stride + kw) - static_cast<long>(padding);
            if (input_col < 0 || input_col >= static_cast<long>(width)) continue;
            const auto input_index = ((sample * channels + channel) * height + static_cast<std::size_t>(input_row)) * width + static_cast<std::size_t>(input_col);
            const float value = input[input_index];
            if (!found || value > best) {
                best = value;
                best_index = input_index;
                found = true;
            }
        }
    }
    if (found) atomicAdd(input_grad + best_index, grad[index]);
}

// Average pooling backward (count-include-padding semantics, matches forward).
__global__ void avg_pool2d_backward_kernel(const float* grad, float* input_grad,
                                           std::size_t batch, std::size_t channels, std::size_t height, std::size_t width,
                                           std::size_t out_h, std::size_t out_w,
                                           std::size_t kernel_size, std::size_t stride, std::size_t padding) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto total = batch * channels * out_h * out_w;
    if (index >= total) return;
    const std::size_t ow = index % out_w;
    const std::size_t oh = (index / out_w) % out_h;
    const std::size_t channel = (index / (out_w * out_h)) % channels;
    const std::size_t sample = index / (channels * out_w * out_h);
    const float share = grad[index] / static_cast<float>(kernel_size * kernel_size);
    for (std::size_t kh = 0; kh < kernel_size; ++kh) {
        const long input_row = static_cast<long>(oh * stride + kh) - static_cast<long>(padding);
        if (input_row < 0 || input_row >= static_cast<long>(height)) continue;
        for (std::size_t kw = 0; kw < kernel_size; ++kw) {
            const long input_col = static_cast<long>(ow * stride + kw) - static_cast<long>(padding);
            if (input_col < 0 || input_col >= static_cast<long>(width)) continue;
            const auto input_index = ((sample * channels + channel) * height + static_cast<std::size_t>(input_row)) * width + static_cast<std::size_t>(input_col);
            atomicAdd(input_grad + input_index, share);
        }
    }
}

// --- tensor elementwise kernels ---
__global__ void tensor_binary_kernel(const float* left, const float* right, float* output,
                                     std::size_t count, bool add) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    output[index] = add ? left[index] + right[index] : left[index] * right[index];
}

__global__ void tensor_scalar_kernel(const float* input, float* output, std::size_t count, float scalar) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    output[index] = input[index] * scalar;
}

void validate_pool(const Tensor& grad, const Tensor& input, std::size_t kernel_size,
                   std::size_t stride, std::size_t padding,
                   std::size_t& out_h, std::size_t& out_w) {
    if (input.rank() != 4 || grad.rank() != 4) throw InvalidArgumentError("pool backward expects rank-4 tensors");
    if (kernel_size == 0 || stride == 0) throw InvalidArgumentError("pool backward kernel/stride must be positive");
    const auto& x = input.shape();
    const auto& g = grad.shape();
    out_h = (x[2] + 2 * padding - kernel_size) / stride + 1;
    out_w = (x[3] + 2 * padding - kernel_size) / stride + 1;
    if (g[0] != x[0] || g[1] != x[1] || g[2] != out_h || g[3] != out_w)
        throw ShapeMismatchError("pool backward gradient shape mismatch");
}

Tensor tensor_elementwise(const Tensor& left, const Tensor& right, bool add, const char* message) {
    if (left.shape() != right.shape()) throw InvalidArgumentError(message);
    Tensor output(left.shape());
    tensor_binary_kernel<<<static_cast<unsigned>((left.size() + 255) / 256), 256, 0, compute_stream()>>>(
        left.device_data(), right.device_data(), output.device_data(), left.size(), add);
    checkCuda(cudaGetLastError(), "tensor elementwise kernel launch");
    output.mark_host_stale();
    return output;
}

}

Tensor conv2d_input_backward(const Tensor& grad, const Tensor& input, const Tensor& weights,
                             std::size_t stride, std::size_t padding) {
    std::size_t out_h = 0, out_w = 0;
    validate_conv(grad, input, weights, out_h, out_w, stride, padding);
    const auto& x = input.shape();
    const auto& w = weights.shape();
    Tensor input_grad(x);
    conv2d_input_backward_kernel<<<static_cast<unsigned>((input_grad.size() + 255) / 256), 256, 0, compute_stream()>>>(
        grad.device_data(), weights.device_data(), input_grad.device_data(),
        x[0], x[1], x[2], x[3], w[0], w[2], w[3], out_h, out_w, stride, padding);
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
    conv2d_weight_backward_kernel<<<static_cast<unsigned>((weight_grad.size() + 255) / 256), 256, 0, compute_stream()>>>(
        grad.device_data(), input.device_data(), weight_grad.device_data(),
        x[0], x[1], x[2], x[3], w[0], w[2], w[3], out_h, out_w, stride, padding);
    checkCuda(cudaGetLastError(), "conv2d weight backward kernel launch");
    weight_grad.mark_host_stale();
    return weight_grad;
}

Tensor conv2d_bias_backward(const Tensor& grad) {
    if (grad.rank() != 4) throw InvalidArgumentError("conv2d bias backward expects a rank-4 gradient");
    const auto& g = grad.shape();
    Tensor bias_grad({g[1]});
    conv2d_bias_backward_kernel<<<static_cast<unsigned>((g[1] + 255) / 256), 256, 0, compute_stream()>>>(
        grad.device_data(), bias_grad.device_data(), g[0], g[1], g[2], g[3]);
    checkCuda(cudaGetLastError(), "conv2d bias backward kernel launch");
    bias_grad.mark_host_stale();
    return bias_grad;
}

Tensor max_pool2d_backward(const Tensor& grad, const Tensor& input,
                           std::size_t kernel_size, std::size_t stride, std::size_t padding) {
    std::size_t out_h = 0, out_w = 0;
    validate_pool(grad, input, kernel_size, stride, padding, out_h, out_w);
    const auto& x = input.shape();
    Tensor input_grad(x);
    // The kernel scatters (atomicAdd) into the buffer, so it must start at zero.
    zero_device_memory(input_grad.device_data(), input_grad.size() * sizeof(float));
    max_pool2d_backward_kernel<<<static_cast<unsigned>((grad.size() + 255) / 256), 256, 0, compute_stream()>>>(
        grad.device_data(), input.device_data(), input_grad.device_data(),
        x[0], x[1], x[2], x[3], out_h, out_w, kernel_size, stride, padding);
    checkCuda(cudaGetLastError(), "max pool backward kernel launch");
    input_grad.mark_host_stale();
    return input_grad;
}

Tensor avg_pool2d_backward(const Tensor& grad, const Tensor& input,
                           std::size_t kernel_size, std::size_t stride, std::size_t padding) {
    std::size_t out_h = 0, out_w = 0;
    validate_pool(grad, input, kernel_size, stride, padding, out_h, out_w);
    const auto& x = input.shape();
    Tensor input_grad(x);
    // The kernel scatters (atomicAdd) into the buffer, so it must start at zero.
    zero_device_memory(input_grad.device_data(), input_grad.size() * sizeof(float));
    avg_pool2d_backward_kernel<<<static_cast<unsigned>((grad.size() + 255) / 256), 256, 0, compute_stream()>>>(
        grad.device_data(), input_grad.device_data(),
        x[0], x[1], x[2], x[3], out_h, out_w, kernel_size, stride, padding);
    checkCuda(cudaGetLastError(), "avg pool backward kernel launch");
    input_grad.mark_host_stale();
    return input_grad;
}

Tensor tensor_add(const Tensor& left, const Tensor& right) {
    return tensor_elementwise(left, right, true, "tensor_add shape mismatch");
}

Tensor tensor_multiply(const Tensor& left, const Tensor& right) {
    return tensor_elementwise(left, right, false, "tensor_multiply shape mismatch");
}

Tensor tensor_multiply(const Tensor& tensor, float scalar) {
    Tensor output(tensor.shape());
    tensor_scalar_kernel<<<static_cast<unsigned>((tensor.size() + 255) / 256), 256, 0, compute_stream()>>>(
        tensor.device_data(), output.device_data(), tensor.size(), scalar);
    checkCuda(cudaGetLastError(), "tensor scalar kernel launch");
    output.mark_host_stale();
    return output;
}

Tensor tensor_negate(const Tensor& tensor) { return tensor_multiply(tensor, -1.0f); }

// batch_matmul backward via cuBLAS strided-batched GEMM.
// grad A = G x B^T (per batch); grad B = A^T x G (per batch).
Tensor batch_matmul_backward_left(const Tensor& grad, const Tensor& right) {
    if (grad.rank() != 3 || right.rank() != 3) throw InvalidArgumentError("batch_matmul backward requires rank-3 tensors");
    const auto& g = grad.shape();
    const auto& b = right.shape();
    if (g[0] != b[0] || g[2] != b[2]) throw ShapeMismatchError("batch_matmul backward shape mismatch");
    Tensor grad_left({g[0], g[1], b[1]});
    if (grad_left.size() == 0) return grad_left;
    const float alpha = 1.0f, beta = 0.0f;
    const auto status = cublasSgemmStridedBatched(
        cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
        static_cast<int>(b[1]), static_cast<int>(g[1]), static_cast<int>(b[2]),
        &alpha, right.device_data(), static_cast<int>(b[2]), static_cast<long long>(b[1] * b[2]),
        grad.device_data(), static_cast<int>(g[2]), static_cast<long long>(g[1] * g[2]),
        &beta, grad_left.device_data(), static_cast<int>(b[1]), static_cast<long long>(g[1] * b[1]),
        static_cast<int>(g[0]));
    if (status != CUBLAS_STATUS_SUCCESS) throw CudaError("cublasSgemmStridedBatched failed");
    grad_left.mark_host_stale();
    return grad_left;
}

Tensor batch_matmul_backward_right(const Tensor& grad, const Tensor& left) {
    if (grad.rank() != 3 || left.rank() != 3) throw InvalidArgumentError("batch_matmul backward requires rank-3 tensors");
    const auto& g = grad.shape();
    const auto& a = left.shape();
    if (g[0] != a[0] || g[1] != a[1]) throw ShapeMismatchError("batch_matmul backward shape mismatch");
    Tensor grad_right({g[0], a[2], g[2]});
    if (grad_right.size() == 0) return grad_right;
    const float alpha = 1.0f, beta = 0.0f;
    const auto status = cublasSgemmStridedBatched(
        cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_T,
        static_cast<int>(g[2]), static_cast<int>(a[2]), static_cast<int>(a[1]),
        &alpha, grad.device_data(), static_cast<int>(g[2]), static_cast<long long>(g[1] * g[2]),
        left.device_data(), static_cast<int>(a[2]), static_cast<long long>(a[1] * a[2]),
        &beta, grad_right.device_data(), static_cast<int>(g[2]), static_cast<long long>(a[2] * g[2]),
        static_cast<int>(g[0]));
    if (status != CUBLAS_STATUS_SUCCESS) throw CudaError("cublasSgemmStridedBatched failed");
    grad_right.mark_host_stale();
    return grad_right;
}

}  // namespace matrix_pro

