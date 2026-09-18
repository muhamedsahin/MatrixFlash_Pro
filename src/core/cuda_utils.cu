#include "matrix_pro/core/cuda_utils.hpp"

#include <cublas_v2.h>
#include <cusolverDn.h>

#ifdef MATRIX_PRO_HAS_CUDNN
#include <cudnn.h>
#endif

#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace matrix_pro {
namespace {

struct CudaExecutionContext {
    int device = 0;
    cudaStream_t stream = nullptr;
    cublasHandle_t blas = nullptr;
    // Persistent cuBLAS workspace: without it cuBLAS allocates/frees an
    // internal scratch buffer on the first GEMM of every new shape, which
    // serializes the stream and fights our own memory pool. One 32 MiB
    // workspace per context removes that cost entirely.
    static constexpr std::size_t workspace_bytes = 32u * 1024u * 1024u;
    void* blas_workspace = nullptr;

    explicit CudaExecutionContext(int device_index) : device(device_index) {
        if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
            throw CudaError("cudaStreamCreateWithFlags failed");
        }
        if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS) {
            cudaStreamDestroy(stream);
            throw CudaError("cublasCreate failed");
        }
        if (cublasSetStream(blas, stream) != CUBLAS_STATUS_SUCCESS) {
            cublasDestroy(blas);
            cudaStreamDestroy(stream);
            throw CudaError("cublasSetStream failed");
        }
        if (cudaMalloc(&blas_workspace, workspace_bytes) == cudaSuccess) {
            if (cublasSetWorkspace(blas, blas_workspace, workspace_bytes) != CUBLAS_STATUS_SUCCESS) {
                cublasSetWorkspace(blas, nullptr, 0);
                cudaFree(blas_workspace);
                blas_workspace = nullptr;
            }
        } else {
            cudaGetLastError(); // clear the sticky error, run without a workspace
        }
        // Allow cuBLAS to use atomic-based split-k reductions; on tall/skinny
        // GEMMs this is measurably faster and only changes accumulation order.
        cublasSetAtomicsMode(blas, CUBLAS_ATOMICS_ALLOWED);
        cudaDeviceProp properties{};
        cudaGetDeviceProperties(&properties, device);
        if (properties.major >= 8) cublasSetMathMode(blas, CUBLAS_TF32_TENSOR_OP_MATH);
    }

    ~CudaExecutionContext() {
        if (blas != nullptr) {
            cublasSetWorkspace(blas, nullptr, 0);
            cublasDestroy(blas);
        }
        if (blas_workspace != nullptr) cudaFree(blas_workspace);
        if (stream != nullptr) cudaStreamDestroy(stream);
    }
};

// One execution context per (host thread, device). A thread that never calls
// into the library allocates nothing, and two threads never share a stream, so
// concurrent use from independent host threads is safe by construction.
struct PerThreadContexts {
    std::unordered_map<int, std::unique_ptr<CudaExecutionContext>> by_device;

    CudaExecutionContext& for_device(int device) {
        auto it = by_device.find(device);
        if (it != by_device.end()) return *it->second;
        auto context = std::make_unique<CudaExecutionContext>(device);
        CudaExecutionContext& reference = *context;
        by_device.emplace(device, std::move(context));
        return reference;
    }

    ~PerThreadContexts() { by_device.clear(); }
};

PerThreadContexts& thread_contexts() {
    static thread_local PerThreadContexts contexts;
    return contexts;
}

CudaExecutionContext& execution_context() {
    int device = 0;
    cudaGetDevice(&device);
    return thread_contexts().for_device(device);
}

// cuSOLVER/cuDNN handles per (thread, device). Lazily created and reused so
// repeated decompositions/convolutions do not pay handle-construction cost,
// while different threads and devices stay isolated from one another.
struct PerThreadSolvers {
    std::unordered_map<int, cusolverDnHandle_t> solver_by_device;
#ifdef MATRIX_PRO_HAS_CUDNN
    std::unordered_map<int, cudnnHandle_t> cudnn_by_device;
#endif

    cusolverDnHandle_t& solver_for(int device) {
        auto it = solver_by_device.find(device);
        if (it != solver_by_device.end()) return it->second;
        cusolverDnHandle_t handle = nullptr;
        if (cusolverDnCreate(&handle) != CUSOLVER_STATUS_SUCCESS) {
            throw SolverError("cuSOLVER handle creation failed");
        }
        if (cusolverDnSetStream(handle, execution_context().stream) != CUSOLVER_STATUS_SUCCESS) {
            cusolverDnDestroy(handle);
            throw SolverError("cusolverDnSetStream failed");
        }
        return solver_by_device.emplace(device, handle).first->second;
    }

#ifdef MATRIX_PRO_HAS_CUDNN
    cudnnHandle_t& cudnn_for(int device) {
        auto it = cudnn_by_device.find(device);
        if (it != cudnn_by_device.end()) return it->second;
        cudnnHandle_t handle = nullptr;
        if (cudnnCreate(&handle) != CUDNN_STATUS_SUCCESS) {
            throw CudaError("cudnnCreate failed");
        }
        if (cudnnSetStream(handle, execution_context().stream) != CUDNN_STATUS_SUCCESS) {
            cudnnDestroy(handle);
            throw CudaError("cudnnSetStream failed");
        }
        return cudnn_by_device.emplace(device, handle).first->second;
    }
#endif

    ~PerThreadSolvers() {
        for (auto& entry : solver_by_device) {
            if (entry.second != nullptr) cusolverDnDestroy(entry.second);
        }
#ifdef MATRIX_PRO_HAS_CUDNN
        for (auto& entry : cudnn_by_device) {
            if (entry.second != nullptr) cudnnDestroy(entry.second);
        }
#endif
    }
};

PerThreadSolvers& thread_solvers() {
    static thread_local PerThreadSolvers solvers;
    return solvers;
}

struct DeviceMemoryPool {
    static constexpr std::size_t max_reuse_per_size = 32;

    struct BlockInfo {
        int device = 0;
        std::size_t size_class = 0;
        // Recorded when the block was recycled (see free_device_memory): a
        // future allocation on a different stream waits on it before reuse.
        cudaEvent_t ready_event = nullptr;
    };

    std::mutex mutex;
    // device -> size class -> reusable blocks
    std::unordered_map<int, std::unordered_map<std::size_t, std::vector<void*>>> free_blocks;
    std::unordered_map<void*, BlockInfo> block_info;

    ~DeviceMemoryPool() {
        for (auto& device_entry : free_blocks) {
            for (auto& bucket : device_entry.second) {
                for (void* ptr : bucket.second) {
                    if (ptr != nullptr) cudaFree(ptr);
                    auto info_it = block_info.find(ptr);
                    if (info_it != block_info.end() && info_it->second.ready_event != nullptr) {
                        cudaEventDestroy(info_it->second.ready_event);
                    }
                }
            }
        }
        for (auto& entry : block_info) {
            if (entry.first != nullptr) cudaFree(entry.first);
        }
    }

    static DeviceMemoryPool& instance() {
        static DeviceMemoryPool pool;
        return pool;
    }
};

// Power-of-two size classes: requests are rounded up so that freed blocks are
// reusable for any nearby request size, avoiding per-exact-size fragmentation.
std::size_t size_class(std::size_t bytes) {
    constexpr std::size_t min_class = 256;
    std::size_t cl = min_class;
    while (cl < bytes) cl *= 2;
    return cl;
}


}

void* allocate_device_memory(std::size_t bytes) {
    if (bytes == 0) return nullptr;
    int device = 0;
    cudaGetDevice(&device);
    auto& pool = DeviceMemoryPool::instance();
    std::lock_guard<std::mutex> lock(pool.mutex);

    const std::size_t cl = size_class(bytes);
    auto& bucket = pool.free_blocks[device][cl];
    if (!bucket.empty()) {
        void* pointer = bucket.back();
        bucket.pop_back();
        const cudaEvent_t ready = pool.block_info[pointer].ready_event;
        pool.block_info[pointer] = DeviceMemoryPool::BlockInfo{device, cl, nullptr};
        if (ready != nullptr) {
            // Cross-stream reuse: make this stream wait until the stream that
            // freed the block (and therefore all GPU work touching it) has
            // reached the recycle point. Same-stream reuse is already ordered
            // by the stream itself, so this only fires across streams.
            const cudaError_t wait_status = cudaStreamWaitEvent(compute_stream(), ready, 0);
            cudaEventDestroy(ready);
            if (wait_status != cudaSuccess) {
                // Could not establish ordering: releasing the block is the
                // only safe fallback (cudaFree implicitly synchronizes).
                pool.block_info.erase(pointer);
                cudaFree(pointer);
                throw CudaError("cudaStreamWaitEvent failed for a recycled memory-pool block");
            }
        }
        return pointer;
    }

    void* pointer = nullptr;
    if (cudaMalloc(&pointer, cl) != cudaSuccess) {
        throw CudaError("cudaMalloc failed in device memory pool");
    }
    pool.block_info[pointer] = DeviceMemoryPool::BlockInfo{device, cl};
    return pointer;
}

void free_device_memory(void* pointer) {
    if (pointer == nullptr) return;
    auto& pool = DeviceMemoryPool::instance();
    std::lock_guard<std::mutex> lock(pool.mutex);

    auto info_it = pool.block_info.find(pointer);
    if (info_it == pool.block_info.end()) {
        cudaFree(pointer);
        return;
    }

    // Recycle into the bucket of the device the block was allocated on, never
    // into another device's bucket.
    const int device = info_it->second.device;
    const std::size_t cl = info_it->second.size_class;
    pool.block_info.erase(info_it);
    auto& bucket = pool.free_blocks[device][cl];
    if (bucket.size() < DeviceMemoryPool::max_reuse_per_size) {
        // Record an event on the current thread's compute stream so a later
        // allocation on a DIFFERENT stream waits for pending GPU work that
        // touched this block before reusing it (see allocate_device_memory).
        // If recording fails for any reason, a plain cudaFree is always safe.
        cudaEvent_t ready = nullptr;
        bool cached = false;
        try {
            if (cudaEventCreateWithFlags(&ready, cudaEventDisableTiming) == cudaSuccess) {
                if (cudaEventRecord(ready, compute_stream()) == cudaSuccess) {
                    pool.block_info[pointer] = DeviceMemoryPool::BlockInfo{device, cl, ready};
                    bucket.push_back(pointer);
                    cached = true;
                }
            }
        } catch (...) {
            // free_device_memory runs from destructors (RAII guards, Matrix) --
            // it must never throw. Fall through to the unconditional free.
        }
        if (cached) return;
        if (ready != nullptr) cudaEventDestroy(ready);
        cudaFree(pointer);
        return;
    }
    cudaFree(pointer);
}

void zero_device_memory(void* pointer, std::size_t bytes) {
    if (pointer == nullptr || bytes == 0) return;
    checkCuda(cudaMemsetAsync(pointer, 0, bytes, compute_stream()), "cudaMemsetAsync in zero_device_memory");
}

void checkCuda(cudaError_t status, const char* operation) {
	if (status != cudaSuccess) {
		throw CudaError(std::string(operation) + ": " + cudaGetErrorString(status));
	}
}

void synchronize() {
    checkCuda(cudaStreamSynchronize(compute_stream()), "cudaStreamSynchronize");
}

int sm_count() {
    int device = 0;
    cudaGetDevice(&device);
    static thread_local std::unordered_map<int, int> cached;
    const auto it = cached.find(device);
    if (it != cached.end()) return it->second;
    int sms = 1;
    if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) != cudaSuccess) {
        cudaGetLastError();
        sms = 1;
    }
    return cached.emplace(device, sms).first->second;
}

cudaStream_t compute_stream() { return execution_context().stream; }
cublasHandle_t& cublas_handle() { return execution_context().blas; }

int device_count() {
    static const int count = [] {
        int detected = 0;
        if (cudaGetDeviceCount(&detected) != cudaSuccess) return 0;
        return detected;
    }();
    return count;
}

int current_device() {
    int device = 0;
    cudaGetDevice(&device);
    return device;
}

void set_device(int device) {
    if (device < 0 || device >= device_count()) {
        throw InvalidArgumentError("Invalid CUDA device index");
    }
    checkCuda(cudaSetDevice(device), "cudaSetDevice");
}

void synchronize_all_devices() {
    for (auto& entry : thread_contexts().by_device) {
        if (entry.second->stream != nullptr) {
            checkCuda(cudaStreamSynchronize(entry.second->stream), "cudaStreamSynchronize (all devices)");
        }
    }
}

DeviceScope::DeviceScope(int device) : previous_device_(current_device()) {
    if (device != previous_device_) set_device(device);
}

DeviceScope::~DeviceScope() {
    if (previous_device_ != current_device()) {
        cudaSetDevice(previous_device_);
    }
}

cusolverDnHandle_t& cusolver_handle() {
    return thread_solvers().solver_for(current_device());
}

#ifdef MATRIX_PRO_HAS_CUDNN
cudnnHandle_t& cudnn_handle() {
    return thread_solvers().cudnn_for(current_device());
}
#endif

}
