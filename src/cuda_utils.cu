#include "matrix_pro/cuda_utils.hpp"

#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace matrix_pro {
namespace {

struct DeviceMemoryPool {
    static constexpr std::size_t max_reuse_per_size = 32;

    std::mutex mutex;
    std::unordered_map<std::size_t, std::vector<void*>> free_blocks;
    std::unordered_map<void*, std::size_t> block_sizes;

    ~DeviceMemoryPool() {
        for (auto& [_, blocks] : free_blocks) {
            for (void* ptr : blocks) {
                if (ptr != nullptr) cudaFree(ptr);
            }
        }
        for (auto& [ptr, _] : block_sizes) {
            if (ptr != nullptr) cudaFree(ptr);
        }
    }

    static DeviceMemoryPool& instance() {
        static DeviceMemoryPool pool;
        return pool;
    }
};

}

void* allocate_device_memory(std::size_t bytes) {
    if (bytes == 0) return nullptr;
    auto& pool = DeviceMemoryPool::instance();
    std::lock_guard<std::mutex> lock(pool.mutex);

    auto& bucket = pool.free_blocks[bytes];
    if (!bucket.empty()) {
        void* pointer = bucket.back();
        bucket.pop_back();
        pool.block_sizes[pointer] = bytes;
        return pointer;
    }

    void* pointer = nullptr;
    if (cudaMalloc(&pointer, bytes) != cudaSuccess) {
        throw std::runtime_error("cudaMalloc failed in device memory pool");
    }
    pool.block_sizes[pointer] = bytes;
    return pointer;
}

void free_device_memory(void* pointer) {
    if (pointer == nullptr) return;
    auto& pool = DeviceMemoryPool::instance();
    std::lock_guard<std::mutex> lock(pool.mutex);

    auto size_it = pool.block_sizes.find(pointer);
    if (size_it == pool.block_sizes.end()) {
        cudaFree(pointer);
        return;
    }

    const std::size_t bytes = size_it->second;
    pool.block_sizes.erase(size_it);
    auto& bucket = pool.free_blocks[bytes];
    if (bucket.size() < DeviceMemoryPool::max_reuse_per_size) {
        bucket.push_back(pointer);
        return;
    }
    cudaFree(pointer);
}

void checkCuda(cudaError_t status, const char* operation) {
	if (status != cudaSuccess) {
		throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
	}
}

void synchronize() {
	checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

}
