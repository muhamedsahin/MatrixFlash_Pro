#pragma once

// Shared elementwise functors for fused kernels (single global-memory pass).

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "matrix_pro/detail/kernel_helpers.cuh"

namespace matrix_pro {
namespace detail {

template <typename Op>
__global__ void ternary_vec4_kernel(const float4* a, const float4* b, const float4* c,
                                    float4* out, std::size_t n4, Op op) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n4) return;
    const float4 x = a[i], y = b[i], z = c[i];
    float4 r;
    r.x = op(x.x, y.x, z.x);
    r.y = op(x.y, y.y, z.y);
    r.z = op(x.z, y.z, z.z);
    r.w = op(x.w, y.w, z.w);
    out[i] = r;
}

template <typename Op>
__global__ void ternary_tail_kernel(const float* a, const float* b, const float* c,
                                    float* out, std::size_t start, std::size_t count, Op op) {
    const std::size_t i = start + static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) out[i] = op(a[i], b[i], c[i]);
}

template <typename Op>
void launch_ternary(const float* a, const float* b, const float* c, float* out,
                    std::size_t count, Op op, cudaStream_t stream) {
    if (count == 0) return;
    std::size_t processed = 0;
    if (count >= 4 && is_aligned4(a) && is_aligned4(b) && is_aligned4(c) && is_aligned4(out)) {
        const std::size_t n4 = count / 4;
        ternary_vec4_kernel<Op><<<grid_blocks(n4), 256, 0, stream>>>(
            reinterpret_cast<const float4*>(a), reinterpret_cast<const float4*>(b),
            reinterpret_cast<const float4*>(c), reinterpret_cast<float4*>(out), n4, op);
        processed = n4 * 4;
    }
    if (processed < count) {
        ternary_tail_kernel<Op><<<grid_blocks(count - processed), 256, 0, stream>>>(
            a, b, c, out, processed, count, op);
    }
}

// Named hot-chain functors (usable with fused_binary / fused_ternary too).
struct SigmoidMulOp {
    __device__ float operator()(float x, float y) const {
        return (1.0f / (1.0f + expf(-x))) * y;
    }
};
struct ReluAddOp {
    __device__ float operator()(float x, float y) const {
        const float s = x + y;
        return s > 0.0f ? s : 0.0f;
    }
};
struct AddMulOp {
    __device__ float operator()(float x, float y, float z) const { return (x + y) * z; }
};
struct ScaleBiasOp {
    float scale; float bias;
    __device__ float operator()(float x) const { return x * scale + bias; }
};
struct BiasGeluOp {
    __device__ float operator()(float x, float b) const {
        const float v = x + b;
        const float x3 = v * v * v;
        return 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * x3)));
    }
};

} // namespace detail
} // namespace matrix_pro
