#include "matrix_pro/core/dtype.hpp"
#include "matrix_pro/core/matrix.hpp"
#include "matrix_pro/core/cuda_utils.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace matrix_pro {
namespace {

__global__ void pack_f16_kernel(const float* src, __half* dst, std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

__global__ void unpack_f16_kernel(const __half* src, float* dst, std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __half2float(src[i]);
}

__global__ void sum_f64_kernel(const float* in, std::size_t n, double* partial) {
    __shared__ double sh[256];
    const unsigned tid = threadIdx.x;
    double acc = 0.0;
    for (std::size_t i = blockIdx.x * 256 + tid; i < n; i += gridDim.x * 256)
        acc += static_cast<double>(in[i]);
    sh[tid] = acc;
    __syncthreads();
    for (unsigned s = 128; s > 0; s >>= 1) {
        if (tid < s) sh[tid] += sh[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = sh[0];
}

__global__ void l2_f64_kernel(const float* in, std::size_t n, double* partial) {
    __shared__ double sh[256];
    const unsigned tid = threadIdx.x;
    double acc = 0.0;
    for (std::size_t i = blockIdx.x * 256 + tid; i < n; i += gridDim.x * 256) {
        const double v = static_cast<double>(in[i]);
        acc += v * v;
    }
    sh[tid] = acc;
    __syncthreads();
    for (unsigned s = 128; s > 0; s >>= 1) {
        if (tid < s) sh[tid] += sh[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = sh[0];
}

double reduce_double_sum(const Matrix& m, bool squared) {
    if (m.empty()) return 0.0;
    const std::size_t n = m.size();
    unsigned blocks = static_cast<unsigned>((n + 255) / 256);
    if (blocks > 256) blocks = 256;
    double* partial = static_cast<double*>(allocate_device_memory(blocks * sizeof(double)));
    if (squared)
        l2_f64_kernel<<<blocks, 256, 0, compute_stream()>>>(m.device_data(), n, partial);
    else
        sum_f64_kernel<<<blocks, 256, 0, compute_stream()>>>(m.device_data(), n, partial);
    checkCuda(cudaGetLastError(), "f64 reduce launch");
    std::vector<double> host(blocks);
    checkCuda(cudaMemcpyAsync(host.data(), partial, blocks * sizeof(double),
                              cudaMemcpyDeviceToHost, compute_stream()), "f64 read");
    checkCuda(cudaStreamSynchronize(compute_stream()), "f64 sync");
    free_device_memory(partial);
    double total = 0.0;
    for (double v : host) total += v;
    return total;
}

} // namespace

template <typename T>
TypedBuffer<T>::TypedBuffer(std::size_t count) : count_(count) {
    if (count_ > 0)
        ptr_ = static_cast<T*>(allocate_device_memory(count_ * sizeof(T)));
}

template <typename T>
TypedBuffer<T>::~TypedBuffer() {
    if (ptr_) free_device_memory(ptr_);
}

template <typename T>
TypedBuffer<T>::TypedBuffer(TypedBuffer&& other) noexcept
    : ptr_(other.ptr_), count_(other.count_) {
    other.ptr_ = nullptr;
    other.count_ = 0;
}

template <typename T>
TypedBuffer<T>& TypedBuffer<T>::operator=(TypedBuffer&& other) noexcept {
    if (this != &other) {
        if (ptr_) free_device_memory(ptr_);
        ptr_ = other.ptr_;
        count_ = other.count_;
        other.ptr_ = nullptr;
        other.count_ = 0;
    }
    return *this;
}

template class TypedBuffer<float>;
template class TypedBuffer<double>;
template class TypedBuffer<std::uint16_t>;

double sum_f64(const Matrix& m) { return reduce_double_sum(m, false); }
double mean_f64(const Matrix& m) {
    if (m.empty()) return 0.0;
    return sum_f64(m) / static_cast<double>(m.size());
}
double l2_norm_f64(const Matrix& m) { return std::sqrt(reduce_double_sum(m, true)); }

void pack_f16(const float* src, void* dst_half, std::size_t count) {
    if (count == 0) return;
    pack_f16_kernel<<<(count + 255) / 256, 256, 0, compute_stream()>>>(
        src, static_cast<__half*>(dst_half), count);
    checkCuda(cudaGetLastError(), "pack_f16 launch");
}

void unpack_f16(const void* src_half, float* dst, std::size_t count) {
    if (count == 0) return;
    unpack_f16_kernel<<<(count + 255) / 256, 256, 0, compute_stream()>>>(
        static_cast<const __half*>(src_half), dst, count);
    checkCuda(cudaGetLastError(), "unpack_f16 launch");
}

} // namespace matrix_pro
