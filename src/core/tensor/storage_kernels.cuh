// Dahili uygulama parçası: src/core/tensor.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Cihaz deleter'ları, fill ve stride-aware kopya çekirdeği.

namespace {
void release_tensor_device(float* pointer) {
    if (pointer != nullptr)
        free_device_memory(pointer);
}

// Shared-storage view deleter — does NOT free the underlying buffer.
void noop_deleter(float*) {
}

__global__ void tensor_fill_kernel(float* values, std::size_t count, float value) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count)
        values[index] = value;
}

// Generic N-D strided copy: src[multi_index ⋅ src_strides + src_offset] → dst[linear]
__global__ void strided_copy_kernel(const float* __restrict__ src, float* __restrict__ dst,
                                    const std::size_t* __restrict__ shape,
                                    const std::size_t* __restrict__ src_strides,
                                    std::size_t src_offset, int rank, std::size_t total) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
         i += stride) {
        // Convert flat index i → N-D multi-index → strided source offset.
        std::size_t src_idx = src_offset;
        std::size_t remaining = i;
        for (int d = rank - 1; d >= 0; --d) {
            const std::size_t dim_idx = remaining % shape[d];
            remaining /= shape[d];
            src_idx += dim_idx * src_strides[d];
        }
        dst[i] = src[src_idx];
    }
}
} // anon
