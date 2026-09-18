#pragma once

// NCHW convolution and pooling over rank-4 tensors, forward and backward.
// A cuDNN backend is used when the toolkit provides it; otherwise the
// hand-written CUDA kernels in src/nn/conv.cu take over -- the API and the
// results are identical either way.

#include <cstddef>

#include "matrix_pro/core/tensor.hpp"

namespace matrix_pro {

// --- forward ---
// Cross-correlation conv2d: input (N,C,H,W), weights (K,C,kH,kW), bias (K).
Tensor conv2d(const Tensor& input, const Tensor& weights, const Tensor& bias,
              std::size_t stride = 1, std::size_t padding = 0);
Tensor max_pool2d(const Tensor& input, std::size_t kernel_size,
                  std::size_t stride = 1, std::size_t padding = 0);
Tensor avg_pool2d(const Tensor& input, std::size_t kernel_size,
                  std::size_t stride = 1, std::size_t padding = 0);

// --- backward ---
// dL/dinput, dL/dweights, dL/dbias for the cross-correlation conv2d above.
Tensor conv2d_input_backward(const Tensor& grad, const Tensor& input, const Tensor& weights,
                             std::size_t stride = 1, std::size_t padding = 0);
Tensor conv2d_weight_backward(const Tensor& grad, const Tensor& input, const Tensor& weights,
                              std::size_t stride = 1, std::size_t padding = 0);
Tensor conv2d_bias_backward(const Tensor& grad);
// dL/dinput for max/avg pooling (avg divides by kernel_size^2, matching the
// forward count-include-padding semantics).
Tensor max_pool2d_backward(const Tensor& grad, const Tensor& input,
                           std::size_t kernel_size, std::size_t stride = 1, std::size_t padding = 0);
Tensor avg_pool2d_backward(const Tensor& grad, const Tensor& input,
                           std::size_t kernel_size, std::size_t stride = 1, std::size_t padding = 0);

} // namespace matrix_pro