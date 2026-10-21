/**
 * @file collectives_cuda.cu
 * @brief Halka all-reduce: her adimda parca P2P kopyalanir, sonra cihaz ustunde toplanir.
 */

#include "matrix_pro/core/errors.hpp"
#include "matrix_pro/detail/cuda_hooks.hpp"

#include <cuda_runtime.h>

#include <vector>

namespace matrix_pro {
namespace hw {
namespace {

__global__ void add_kernel(float* dst, const float* src, std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        dst[index] += src[index];
    }
}

__global__ void max_kernel(float* dst, const float* src, std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        dst[index] = dst[index] > src[index] ? dst[index] : src[index];
    }
}

__global__ void scale_kernel(float* data, std::size_t count, float scale) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        data[index] *= scale;
    }
}

}  // namespace

void cuda_launch_add(int device, float* dst, const float* src, std::size_t count, cudaStream_t stream) {
    cudaSetDevice(device);
    const int threads = 256;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    add_kernel<<<blocks, threads, 0, stream>>>(dst, src, count);
    if (cudaGetLastError() != cudaSuccess) {
        throw MatrixProError("CUDA add kernel failed");
    }
}

void cuda_launch_max(int device, float* dst, const float* src, std::size_t count, cudaStream_t stream) {
    cudaSetDevice(device);
    const int threads = 256;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    max_kernel<<<blocks, threads, 0, stream>>>(dst, src, count);
    if (cudaGetLastError() != cudaSuccess) {
        throw MatrixProError("CUDA max kernel failed");
    }
}

void cuda_launch_scale(int device, float* data, std::size_t count, float scale, cudaStream_t stream) {
    cudaSetDevice(device);
    const int threads = 256;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    scale_kernel<<<blocks, threads, 0, stream>>>(data, count, scale);
    if (cudaGetLastError() != cudaSuccess) {
        throw MatrixProError("CUDA scale kernel failed");
    }
}

void cuda_allreduce(float** data, const int* devices, int rank_count, std::size_t count, int reduce_op, int) {
    if (rank_count <= 0 || data == nullptr) {
        throw MatrixProError("cuda allreduce: empty");
    }
    if (rank_count == 1) {
        if (reduce_op == 1) {
            cuda_scale_inplace(devices[0], data[0], count, 1.0f);
            cuda_sync(devices[0]);
        }
        return;
    }
    const std::size_t chunk = count / static_cast<std::size_t>(rank_count);
    if (chunk == 0 || count % static_cast<std::size_t>(rank_count) != 0) {
        throw MatrixProError("cuda ring allreduce: length must be divisible by rank count");
    }
    std::vector<float*> scratch(static_cast<std::size_t>(rank_count));
    for (int rank = 0; rank < rank_count; ++rank) {
        scratch[static_cast<std::size_t>(rank)] = cuda_alloc(devices[rank], chunk);
    }
    for (int step = 0; step < rank_count - 1; ++step) {
        for (int rank = 0; rank < rank_count; ++rank) {
            const int send_index = (rank - step + rank_count) % rank_count;
            const int destination = (rank + 1) % rank_count;
            const std::size_t offset = static_cast<std::size_t>(send_index) * chunk;
            cuda_copy_p2p(devices[destination], scratch[static_cast<std::size_t>(destination)], devices[rank],
                          data[rank] + offset, chunk);
        }
        for (int rank = 0; rank < rank_count; ++rank) {
            cuda_sync(devices[rank]);
        }
        for (int rank = 0; rank < rank_count; ++rank) {
            const int source = (rank - 1 + rank_count) % rank_count;
            const int send_index = (source - step + rank_count) % rank_count;
            const std::size_t offset = static_cast<std::size_t>(send_index) * chunk;
            if (reduce_op == 2) {
                cuda_max_inplace(devices[rank], data[rank] + offset, scratch[static_cast<std::size_t>(rank)], chunk);
            } else {
                cuda_add_inplace(devices[rank], data[rank] + offset, scratch[static_cast<std::size_t>(rank)], chunk);
            }
        }
        for (int rank = 0; rank < rank_count; ++rank) {
            cuda_sync(devices[rank]);
        }
    }
    for (int step = 0; step < rank_count - 1; ++step) {
        for (int rank = 0; rank < rank_count; ++rank) {
            const int send_index = (rank - step + 1 + rank_count) % rank_count;
            const int destination = (rank + 1) % rank_count;
            const std::size_t offset = static_cast<std::size_t>(send_index) * chunk;
            cuda_copy_p2p(devices[destination], scratch[static_cast<std::size_t>(destination)], devices[rank],
                          data[rank] + offset, chunk);
        }
        for (int rank = 0; rank < rank_count; ++rank) {
            cuda_sync(devices[rank]);
        }
        for (int rank = 0; rank < rank_count; ++rank) {
            const int source = (rank - 1 + rank_count) % rank_count;
            const int send_index = (source - step + 1 + rank_count) % rank_count;
            const std::size_t offset = static_cast<std::size_t>(send_index) * chunk;
            cuda_copy_p2p(devices[rank], data[rank] + offset, devices[rank], scratch[static_cast<std::size_t>(rank)], chunk);
        }
        for (int rank = 0; rank < rank_count; ++rank) {
            cuda_sync(devices[rank]);
        }
    }
    if (reduce_op == 1) {
        const float scale = 1.0f / static_cast<float>(rank_count);
        for (int rank = 0; rank < rank_count; ++rank) {
            cuda_scale_inplace(devices[rank], data[rank], count, scale);
            cuda_sync(devices[rank]);
        }
    }
    for (int rank = 0; rank < rank_count; ++rank) {
        cuda_free(devices[rank], scratch[static_cast<std::size_t>(rank)]);
    }
}

}  // namespace hw
}  // namespace matrix_pro
