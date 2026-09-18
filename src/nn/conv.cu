#include "matrix_pro/ops/operations.hpp"
#include "matrix_pro/core/tensor.hpp"
#include "matrix_pro/core/cuda_utils.hpp"

#include <cfloat>
#include <string>
#include <stdexcept>

#ifdef MATRIX_PRO_HAS_CUDNN
#include <cudnn.h>
#endif

namespace matrix_pro {
namespace {

#ifdef MATRIX_PRO_HAS_CUDNN
void check_cudnn(cudnnStatus_t status, const char* operation) {
    if (status != CUDNN_STATUS_SUCCESS) throw CudaError(std::string(operation) + ": " + cudnnGetErrorString(status));
}
#endif

__global__ void conv2d_kernel(const float* input, const float* weights, const float* bias, float* output,
                              std::size_t batch, std::size_t channels, std::size_t height, std::size_t width,
                              std::size_t filters, std::size_t kernel_height, std::size_t kernel_width,
                              std::size_t output_height, std::size_t output_width,
                              std::size_t stride, std::size_t padding) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto total = batch * filters * output_height * output_width;
    if (index >= total) return;
    const std::size_t ow = index % output_width;
    const std::size_t oh = (index / output_width) % output_height;
    const std::size_t filter = (index / (output_width * output_height)) % filters;
    const std::size_t sample = index / (filters * output_width * output_height);
    float sum = bias[filter];
    for (std::size_t channel = 0; channel < channels; ++channel) {
        for (std::size_t kh = 0; kh < kernel_height; ++kh) {
            const long input_row = static_cast<long>(oh * stride + kh) - static_cast<long>(padding);
            if (input_row < 0 || input_row >= static_cast<long>(height)) continue;
            for (std::size_t kw = 0; kw < kernel_width; ++kw) {
                const long input_col = static_cast<long>(ow * stride + kw) - static_cast<long>(padding);
                if (input_col < 0 || input_col >= static_cast<long>(width)) continue;
                const auto input_index = ((sample * channels + channel) * height + static_cast<std::size_t>(input_row)) * width + static_cast<std::size_t>(input_col);
                const auto weight_index = ((filter * channels + channel) * kernel_height + kh) * kernel_width + kw;
                sum += input[input_index] * weights[weight_index];
            }
        }
    }
    output[index] = sum;
}

__global__ void pool2d_kernel(const float* input, float* output,
                              std::size_t batch, std::size_t channels, std::size_t height, std::size_t width,
                              std::size_t output_height, std::size_t output_width,
                              std::size_t kernel_size, std::size_t stride, std::size_t padding, bool maximum) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto total = batch * channels * output_height * output_width;
    if (index >= total) return;
    const std::size_t ow = index % output_width;
    const std::size_t oh = (index / output_width) % output_height;
    const std::size_t channel = (index / (output_width * output_height)) % channels;
    const std::size_t sample = index / (channels * output_width * output_height);
    float result = maximum ? -FLT_MAX : 0.0f;
    for (std::size_t kh = 0; kh < kernel_size; ++kh) {
        const long input_row = static_cast<long>(oh * stride + kh) - static_cast<long>(padding);
        if (input_row < 0 || input_row >= static_cast<long>(height)) continue;
        for (std::size_t kw = 0; kw < kernel_size; ++kw) {
            const long input_col = static_cast<long>(ow * stride + kw) - static_cast<long>(padding);
            if (input_col < 0 || input_col >= static_cast<long>(width)) continue;
            const auto input_index = ((sample * channels + channel) * height + static_cast<std::size_t>(input_row)) * width + static_cast<std::size_t>(input_col);
            const float value = input[input_index];
            if (maximum) result = fmaxf(result, value);
            else result += value;
        }
    }
    output[index] = maximum ? result : result / static_cast<float>(kernel_size * kernel_size);
}

void validate_nchw(const Tensor& input) {
    if (input.rank() != 4) throw ShapeMismatchError("CNN input must have shape [N,C,H,W]");
}

#ifdef MATRIX_PRO_HAS_CUDNN
Tensor cudnn_pool2d(const Tensor& input, std::size_t kernel_size, std::size_t stride,
                    std::size_t padding, bool maximum) {
    const auto& x = input.shape();
    Tensor output({x[0], x[1], (x[2] + 2 * padding - kernel_size) / stride + 1, (x[3] + 2 * padding - kernel_size) / stride + 1});
    cudnnTensorDescriptor_t input_desc = nullptr, output_desc = nullptr;
    cudnnPoolingDescriptor_t pooling_desc = nullptr;
    try {
        check_cudnn(cudnnCreateTensorDescriptor(&input_desc), "pool input descriptor");
        check_cudnn(cudnnCreateTensorDescriptor(&output_desc), "pool output descriptor");
        check_cudnn(cudnnCreatePoolingDescriptor(&pooling_desc), "pool descriptor");
        check_cudnn(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, static_cast<int>(x[0]), static_cast<int>(x[1]), static_cast<int>(x[2]), static_cast<int>(x[3])), "pool input setup");
        check_cudnn(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, static_cast<int>(output.shape()[0]), static_cast<int>(output.shape()[1]), static_cast<int>(output.shape()[2]), static_cast<int>(output.shape()[3])), "pool output setup");
        check_cudnn(cudnnSetPooling2dDescriptor(pooling_desc, maximum ? CUDNN_POOLING_MAX : CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, CUDNN_NOT_PROPAGATE_NAN, static_cast<int>(kernel_size), static_cast<int>(kernel_size), static_cast<int>(padding), static_cast<int>(padding), static_cast<int>(stride), static_cast<int>(stride)), "pool descriptor setup");
        const float alpha = 1.0f, beta = 0.0f;
        check_cudnn(cudnnPoolingForward(cudnn_handle(), pooling_desc, &alpha, input_desc, input.device_data(), &beta, output_desc, output.device_data()), "cudnnPoolingForward");
        cudnnDestroyPoolingDescriptor(pooling_desc); cudnnDestroyTensorDescriptor(output_desc); cudnnDestroyTensorDescriptor(input_desc);
        output.mark_host_stale();
        return output;
    } catch (...) {
        if (pooling_desc) cudnnDestroyPoolingDescriptor(pooling_desc); if (output_desc) cudnnDestroyTensorDescriptor(output_desc); if (input_desc) cudnnDestroyTensorDescriptor(input_desc);
        throw;
    }
}
#endif

}

Tensor conv2d(const Tensor& input, const Tensor& weights, const Tensor& bias,
              std::size_t stride, std::size_t padding) {
    validate_nchw(input);
    if (weights.rank() != 4 || bias.rank() != 1) throw InvalidArgumentError("conv2d expects weights [K,C,R,S] and bias [K]");
    const auto& x = input.shape();
    const auto& w = weights.shape();
    if (w[1] != x[1] || bias.shape()[0] != w[0] || stride == 0) throw ShapeMismatchError("conv2d channel or stride mismatch");
    if (x[2] + 2 * padding < w[2] || x[3] + 2 * padding < w[3]) throw ShapeMismatchError("conv2d kernel is larger than padded input");
    const std::size_t output_height = (x[2] + 2 * padding - w[2]) / stride + 1;
    const std::size_t output_width = (x[3] + 2 * padding - w[3]) / stride + 1;
    Tensor output({x[0], w[0], output_height, output_width});
#ifdef MATRIX_PRO_HAS_CUDNN
    cudnnTensorDescriptor_t input_desc = nullptr, output_desc = nullptr, bias_desc = nullptr;
    cudnnFilterDescriptor_t filter_desc = nullptr;
    cudnnConvolutionDescriptor_t convolution_desc = nullptr;
    void* workspace = nullptr;
    try {
        check_cudnn(cudnnCreateTensorDescriptor(&input_desc), "input descriptor");
        check_cudnn(cudnnCreateTensorDescriptor(&output_desc), "output descriptor");
        check_cudnn(cudnnCreateTensorDescriptor(&bias_desc), "bias descriptor");
        check_cudnn(cudnnCreateFilterDescriptor(&filter_desc), "filter descriptor");
        check_cudnn(cudnnCreateConvolutionDescriptor(&convolution_desc), "convolution descriptor");
        check_cudnn(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, static_cast<int>(x[0]), static_cast<int>(x[1]), static_cast<int>(x[2]), static_cast<int>(x[3])), "input descriptor setup");
        check_cudnn(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, static_cast<int>(x[0]), static_cast<int>(w[0]), static_cast<int>(output_height), static_cast<int>(output_width)), "output descriptor setup");
        check_cudnn(cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, static_cast<int>(w[0]), 1, 1), "bias descriptor setup");
        check_cudnn(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, static_cast<int>(w[0]), static_cast<int>(w[1]), static_cast<int>(w[2]), static_cast<int>(w[3])), "filter descriptor setup");
        check_cudnn(cudnnSetConvolution2dDescriptor(convolution_desc, static_cast<int>(padding), static_cast<int>(padding), static_cast<int>(stride), static_cast<int>(stride), 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT), "convolution descriptor setup");
        cudnnConvolutionFwdAlgo_t algorithm = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
        std::size_t workspace_size = 0;
        check_cudnn(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle(), input_desc, filter_desc, convolution_desc, output_desc, algorithm, &workspace_size), "convolution workspace query");
        if (workspace_size != 0) checkCuda(cudaMalloc(&workspace, workspace_size), "cuDNN workspace allocation");
        const float alpha = 1.0f, beta = 0.0f;
        check_cudnn(cudnnConvolutionForward(cudnn_handle(), &alpha, input_desc, input.device_data(), filter_desc, weights.device_data(), convolution_desc, algorithm, workspace, workspace_size, &beta, output_desc, output.device_data()), "cudnnConvolutionForward");
        check_cudnn(cudnnAddTensor(cudnn_handle(), &alpha, bias_desc, bias.device_data(), &alpha, output_desc, output.device_data()), "cudnnAddTensor");
        if (workspace) cudaFree(workspace);
        cudnnDestroyConvolutionDescriptor(convolution_desc); cudnnDestroyFilterDescriptor(filter_desc);
        cudnnDestroyTensorDescriptor(bias_desc); cudnnDestroyTensorDescriptor(output_desc); cudnnDestroyTensorDescriptor(input_desc);
        output.mark_host_stale();
        return output;
    } catch (...) {
        if (workspace) cudaFree(workspace);
        if (convolution_desc) cudnnDestroyConvolutionDescriptor(convolution_desc); if (filter_desc) cudnnDestroyFilterDescriptor(filter_desc);
        if (bias_desc) cudnnDestroyTensorDescriptor(bias_desc); if (output_desc) cudnnDestroyTensorDescriptor(output_desc); if (input_desc) cudnnDestroyTensorDescriptor(input_desc);
        throw;
    }
#else
    conv2d_kernel<<<static_cast<unsigned>((output.size() + 255) / 256), 256, 0, compute_stream()>>>(
        input.device_data(), weights.device_data(), bias.device_data(), output.device_data(),
        x[0], x[1], x[2], x[3], w[0], w[2], w[3], output_height, output_width, stride, padding);
    checkCuda(cudaGetLastError(), "conv2d kernel launch");
    output.mark_host_stale();
    return output;
#endif
}

Tensor max_pool2d(const Tensor& input, std::size_t kernel_size, std::size_t stride, std::size_t padding) {
    validate_nchw(input);
    if (kernel_size == 0 || stride == 0) throw InvalidArgumentError("pooling kernel and stride must be positive");
    const auto& x = input.shape();
    if (x[2] + 2 * padding < kernel_size || x[3] + 2 * padding < kernel_size) throw ShapeMismatchError("pooling kernel is larger than padded input");
#ifdef MATRIX_PRO_HAS_CUDNN
    return cudnn_pool2d(input, kernel_size, stride, padding, true);
#else
    Tensor output({x[0], x[1], (x[2] + 2 * padding - kernel_size) / stride + 1, (x[3] + 2 * padding - kernel_size) / stride + 1});
    pool2d_kernel<<<static_cast<unsigned>((output.size() + 255) / 256), 256, 0, compute_stream()>>>(input.device_data(), output.device_data(), x[0], x[1], x[2], x[3], output.shape()[2], output.shape()[3], kernel_size, stride, padding, true);
    checkCuda(cudaGetLastError(), "max pool kernel launch");
    output.mark_host_stale();
    return output;
#endif
}

Tensor avg_pool2d(const Tensor& input, std::size_t kernel_size, std::size_t stride, std::size_t padding) {
    validate_nchw(input);
    if (kernel_size == 0 || stride == 0) throw InvalidArgumentError("pooling kernel and stride must be positive");
    const auto& x = input.shape();
    if (x[2] + 2 * padding < kernel_size || x[3] + 2 * padding < kernel_size) throw ShapeMismatchError("pooling kernel is larger than padded input");
#ifdef MATRIX_PRO_HAS_CUDNN
    return cudnn_pool2d(input, kernel_size, stride, padding, false);
#else
    Tensor output({x[0], x[1], (x[2] + 2 * padding - kernel_size) / stride + 1, (x[3] + 2 * padding - kernel_size) / stride + 1});
    pool2d_kernel<<<static_cast<unsigned>((output.size() + 255) / 256), 256, 0, compute_stream()>>>(input.device_data(), output.device_data(), x[0], x[1], x[2], x[3], output.shape()[2], output.shape()[3], kernel_size, stride, padding, false);
    checkCuda(cudaGetLastError(), "avg pool kernel launch");
    output.mark_host_stale();
    return output;
#endif
}
}
