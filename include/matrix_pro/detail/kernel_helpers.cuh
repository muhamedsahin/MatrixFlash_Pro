#pragma once

// Internal helpers shared by the CUDA translation units: 16-byte (float4)
// vectorized elementwise launch utilities. Included only from .cu files, so the
// CUDA types never leak into the public C++ headers.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace matrix_pro {
namespace detail {

inline unsigned grid_blocks(std::size_t count, unsigned block = 256) {
    return static_cast<unsigned>((count + block - 1) / block);
}

// True when a device pointer is 16-byte aligned, which is required before it can
// be reinterpreted as float4*. Buffers coming from the device memory pool are
// cudaMalloc-backed (256-byte aligned), so this holds for base pointers.
inline bool is_aligned4(const void* pointer) {
    return (reinterpret_cast<std::uintptr_t>(pointer) & 0xFu) == 0u;
}

// --- grid sizing ------------------------------------------------------------
// Caps the grid so partial buffers and launch overhead stay small while every
// SM still gets several waves of blocks. Kernels written as grid-stride loops
// cover any element count with this grid.
inline unsigned capped_grid(std::size_t work_items, unsigned block) {
    int device = 0;
    cudaGetDevice(&device);
    int sms = 0;
    if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) != cudaSuccess) {
        cudaGetLastError();
        sms = 1;
    }
    constexpr unsigned blocks_per_sm = 8;
    const unsigned max_blocks = static_cast<unsigned>(sms) * blocks_per_sm;
    const unsigned needed = grid_blocks(work_items, block);
    return needed < max_blocks ? needed : max_blocks;
}

// --- generic two-input elementwise -----------------------------------------
template <typename Op>
__global__ void binary_vec4_kernel(const float4* __restrict__ a, const float4* __restrict__ b,
                                   float4* __restrict__ out, std::size_t n4, Op op) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n4; i += stride) {
        const float4 x = a[i];
        const float4 y = b[i];
        float4 r;
        r.x = op(x.x, y.x);
        r.y = op(x.y, y.y);
        r.z = op(x.z, y.z);
        r.w = op(x.w, y.w);
        out[i] = r;
    }
}

template <typename Op>
__global__ void binary_tail_kernel(const float* __restrict__ a, const float* __restrict__ b,
                                   float* __restrict__ out, std::size_t count, Op op) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += stride) {
        out[i] = op(a[i], b[i]);
    }
}

// Processes the aligned float4 prefix and a scalar tail in a single call.
template <typename Op>
void launch_binary(const float* a, const float* b, float* out, std::size_t count,
                   Op op, cudaStream_t stream) {
    if (count == 0) return;
    std::size_t processed = 0;
    if (count >= 4 && is_aligned4(a) && is_aligned4(b) && is_aligned4(out)) {
        const std::size_t n4 = count / 4;
        binary_vec4_kernel<Op><<<capped_grid(n4, 256), 256, 0, stream>>>(
            reinterpret_cast<const float4*>(a), reinterpret_cast<const float4*>(b),
            reinterpret_cast<float4*>(out), n4, op);
        processed = n4 * 4;
    }
    if (processed < count) {
        binary_tail_kernel<Op><<<capped_grid(count - processed, 256), 256, 0, stream>>>(
            a + processed, b + processed, out + processed, count - processed, op);
    }
}

// --- generic one-input elementwise -----------------------------------------
template <typename Op>
__global__ void unary_vec4_kernel(const float4* __restrict__ in, float4* __restrict__ out,
                                  std::size_t n4, Op op) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n4; i += stride) {
        const float4 x = in[i];
        float4 r;
        r.x = op(x.x);
        r.y = op(x.y);
        r.z = op(x.z);
        r.w = op(x.w);
        out[i] = r;
    }
}

template <typename Op>
__global__ void unary_tail_kernel(const float* __restrict__ in, float* __restrict__ out,
                                  std::size_t count, Op op) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += stride) {
        out[i] = op(in[i]);
    }
}

template <typename Op>
void launch_unary(const float* in, float* out, std::size_t count, Op op, cudaStream_t stream) {
    if (count == 0) return;
    std::size_t processed = 0;
    if (count >= 4 && is_aligned4(in) && is_aligned4(out)) {
        const std::size_t n4 = count / 4;
        unary_vec4_kernel<Op><<<capped_grid(n4, 256), 256, 0, stream>>>(
            reinterpret_cast<const float4*>(in), reinterpret_cast<float4*>(out), n4, op);
        processed = n4 * 4;
    }
    if (processed < count) {
        unary_tail_kernel<Op><<<capped_grid(count - processed, 256), 256, 0, stream>>>(
            in + processed, out + processed, count - processed, op);
    }
}

} // namespace detail
} // namespace matrix_pro