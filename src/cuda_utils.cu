#include "matrix_pro/cuda_utils.hpp"

#include <cublas_v2.h>
#include <cusolverDn.h>

#ifdef MATRIX_PRO_HAS_CUDNN
#include <cudnn.h>
#endif

#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace matrix_pro {
namespace {

struct CudaExecutionContext {
    cudaStream_t stream = nullptr;
    cublasHandle_t blas = nullptr;

    CudaExecutionContext() {
        if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
            throw std::runtime_error("cudaStreamCreateWithFlags failed");
        }
        if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS) {
            cudaStreamDestroy(stream);
            throw std::runtime_error("cublasCreate failed");
        }
        if (cublasSetStream(blas, stream) != CUBLAS_STATUS_SUCCESS) {
            cublasDestroy(blas);
            cudaStreamDestroy(stream);
            throw std::runtime_error("cublasSetStream failed");
        }
        int device = 0;
        cudaGetDevice(&device);
        cudaDeviceProp properties{};
        cudaGetDeviceProperties(&properties, device);
        if (properties.major >= 8) cublasSetMathMode(blas, CUBLAS_TF32_TENSOR_OP_MATH);
    }

    ~CudaExecutionContext() {
        if (blas != nullptr) cublasDestroy(blas);
        if (stream != nullptr) cudaStreamDestroy(stream);
    }
};

CudaExecutionContext& execution_context() {
    static CudaExecutionContext context;
    return context;
}

struct CusolverHandleGuard {
    cusolverDnHandle_t handle = nullptr;

    CusolverHandleGuard() {
        if (cusolverDnCreate(&handle) != CUSOLVER_STATUS_SUCCESS) {
            throw std::runtime_error("cuSOLVER handle creation failed");
        }
        if (cusolverDnSetStream(handle, execution_context().stream) != CUSOLVER_STATUS_SUCCESS) {
            cusolverDnDestroy(handle);
            throw std::runtime_error("cusolverDnSetStream failed");
        }
    }

    ~CusolverHandleGuard() {
        if (handle != nullptr) {
            cusolverDnDestroy(handle);
        }
    }
};

#ifdef MATRIX_PRO_HAS_CUDNN
struct CudnnHandleGuard {
    cudnnHandle_t handle = nullptr;

    CudnnHandleGuard() {
        if (cudnnCreate(&handle) != CUDNN_STATUS_SUCCESS) {
            throw std::runtime_error("cudnnCreate failed");
        }
        if (cudnnSetStream(handle, execution_context().stream) != CUDNN_STATUS_SUCCESS) {
            cudnnDestroy(handle);
            throw std::runtime_error("cudnnSetStream failed");
        }
    }

    ~CudnnHandleGuard() {
        if (handle != nullptr) cudnnDestroy(handle);
    }
};
#endif

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
    checkCuda(cudaStreamSynchronize(compute_stream()), "cudaStreamSynchronize");
}

cudaStream_t compute_stream() { return execution_context().stream; }
cublasHandle_t& cublas_handle() { return execution_context().blas; }

cusolverDnHandle_t& cusolver_handle() {
    static CusolverHandleGuard guard;
    return guard.handle;
}

#ifdef MATRIX_PRO_HAS_CUDNN
cudnnHandle_t& cudnn_handle() {
    static CudnnHandleGuard guard;
    return guard.handle;
}
#endif

}
