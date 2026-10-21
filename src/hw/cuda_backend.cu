/**
 * @file cuda_backend.cu
 * @brief FP32 pedantic cuBLAS GEMM ve aygit kopyalari. Akis senkronize edilmez;
 * cagiran cuda_sync ile bekler, boylece CPU ile ortusur.
 */

#include "matrix_pro/core/backend_registry.hpp"
#include "matrix_pro/core/errors.hpp"
#include "matrix_pro/detail/cuda_hooks.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace matrix_pro {
namespace hw {
namespace {

struct Slot {
    cudaStream_t stream = nullptr;
    cublasHandle_t handle = nullptr;
    std::mutex mutex;
};

std::mutex& table_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::vector<std::unique_ptr<Slot>>& slots() {
    static std::vector<std::unique_ptr<Slot>> table;
    return table;
}

void check_cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw MatrixProError(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

void check_blas(cublasStatus_t status, const char* what) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw MatrixProError(std::string(what) + " failed");
    }
}

Slot& slot(int device) {
    std::lock_guard<std::mutex> lock(table_mutex());
    if (device < 0 || device >= static_cast<int>(slots().size()) || slots()[static_cast<std::size_t>(device)] == nullptr) {
        throw MatrixProError("CUDA device index out of range");
    }
    Slot& entry = *slots()[static_cast<std::size_t>(device)];
    if (entry.stream == nullptr) {
        check_cuda(cudaSetDevice(device), "cudaSetDevice");
        check_cuda(cudaStreamCreate(&entry.stream), "cudaStreamCreate");
        check_blas(cublasCreate(&entry.handle), "cublasCreate");
        check_blas(cublasSetStream(entry.handle, entry.stream), "cublasSetStream");
        check_blas(cublasSetMathMode(entry.handle, CUBLAS_PEDANTIC_MATH), "cublasSetMathMode");
    }
    return entry;
}

void ensure_table() {
    std::lock_guard<std::mutex> lock(table_mutex());
    if (!slots().empty()) {
        return;
    }
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 0) {
        count = 0;
    }
    slots().clear();
    for (int i = 0; i < count; ++i) {
        slots().emplace_back(new Slot());
        (void)i;
    }
}

}  // namespace

int cuda_device_count() {
    ensure_table();
    return static_cast<int>(slots().size());
}

void cuda_gemm(int device, int m, int n, int k, const float* a, int lda, const float* b, int ldb, float* c, int ldc,
               float alpha, float beta) {
    Slot& entry = slot(device);
    std::lock_guard<std::mutex> lock(entry.mutex);
    check_cuda(cudaSetDevice(device), "cudaSetDevice");
    // Row-major C = A*B  <=>  column-major C^T = B^T * A^T, issued as N,N with B then A.
    check_blas(cublasSgemm(entry.handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, b, ldb, a, lda, &beta, c, ldc),
               "cublasSgemm");
}

void cuda_copy_h2d(int device, float* dst, const float* src, std::size_t count) {
    Slot& entry = slot(device);
    std::lock_guard<std::mutex> lock(entry.mutex);
    check_cuda(cudaSetDevice(device), "cudaSetDevice");
    check_cuda(cudaMemcpyAsync(dst, src, count * sizeof(float), cudaMemcpyHostToDevice, entry.stream), "H2D");
}

void cuda_copy_d2h(int device, float* dst, const float* src, std::size_t count) {
    Slot& entry = slot(device);
    std::lock_guard<std::mutex> lock(entry.mutex);
    check_cuda(cudaSetDevice(device), "cudaSetDevice");
    check_cuda(cudaMemcpyAsync(dst, src, count * sizeof(float), cudaMemcpyDeviceToHost, entry.stream), "D2H");
}

void cuda_copy_p2p(int dst_device, float* dst, int src_device, const float* src, std::size_t count) {
    Slot& entry = slot(dst_device);
    std::lock_guard<std::mutex> lock(entry.mutex);
    check_cuda(cudaMemcpyPeerAsync(dst, dst_device, src, src_device, count * sizeof(float), entry.stream), "P2P");
}

void cuda_add_inplace(int device, float* dst, const float* src, std::size_t count) {
    // Launched as a small grid. Defined in collectives_cuda.cu to keep one kernel TU.
    extern void cuda_launch_add(int device, float* dst, const float* src, std::size_t count, cudaStream_t stream);
    Slot& entry = slot(device);
    std::lock_guard<std::mutex> lock(entry.mutex);
    cuda_launch_add(device, dst, src, count, entry.stream);
}

void cuda_max_inplace(int device, float* dst, const float* src, std::size_t count) {
    extern void cuda_launch_max(int device, float* dst, const float* src, std::size_t count, cudaStream_t stream);
    Slot& entry = slot(device);
    std::lock_guard<std::mutex> lock(entry.mutex);
    cuda_launch_max(device, dst, src, count, entry.stream);
}

void cuda_scale_inplace(int device, float* data, std::size_t count, float scale) {
    extern void cuda_launch_scale(int device, float* data, std::size_t count, float scale, cudaStream_t stream);
    Slot& entry = slot(device);
    std::lock_guard<std::mutex> lock(entry.mutex);
    cuda_launch_scale(device, data, count, scale, entry.stream);
}

float* cuda_alloc(int device, std::size_t count) {
    std::lock_guard<std::mutex> lock(slot(device).mutex);
    check_cuda(cudaSetDevice(device), "cudaSetDevice");
    float* pointer = nullptr;
    check_cuda(cudaMalloc(&pointer, count * sizeof(float)), "cudaMalloc");
    return pointer;
}

void cuda_free(int device, float* pointer) {
    if (pointer == nullptr) {
        return;
    }
    cudaSetDevice(device);
    cudaFree(pointer);
}

void cuda_sync(int device) {
    Slot& entry = slot(device);
    std::lock_guard<std::mutex> lock(entry.mutex);
    check_cuda(cudaStreamSynchronize(entry.stream), "cudaStreamSynchronize");
}

bool cuda_peer_access(int a, int b) {
    int can = 0;
    if (cudaDeviceCanAccessPeer(&can, a, b) != cudaSuccess) {
        return false;
    }
    return can != 0;
}

bool cuda_nvlink(int a, int b) {
    int native_atomics = 0;
    if (cudaDeviceGetP2PAttribute(&native_atomics, cudaDevP2PAttrNativeAtomicSupported, a, b) != cudaSuccess) {
        return false;
    }
    return native_atomics != 0;
}

void cuda_enable_peer(int a, int b) {
    if (!cuda_peer_access(a, b)) {
        return;
    }
    check_cuda(cudaSetDevice(a), "cudaSetDevice");
    const cudaError_t status = cudaDeviceEnablePeerAccess(b, 0);
    if (status != cudaSuccess && status != cudaErrorPeerAccessAlreadyEnabled) {
        throw MatrixProError(std::string("cudaDeviceEnablePeerAccess: ") + cudaGetErrorString(status));
    }
    cudaGetLastError();
}

const char* cuda_device_name(int device) {
    static thread_local std::string storage;
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
        return "cuda";
    }
    storage = prop.name;
    return storage.c_str();
}

std::uint64_t cuda_memory_bytes(int device) {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
        return 0;
    }
    return static_cast<std::uint64_t>(prop.totalGlobalMem);
}

double cuda_measure_gflops(int device) {
    constexpr int kN = 512;
    const std::size_t cells = static_cast<std::size_t>(kN) * kN;
    std::vector<float> host(cells, 0.01f);
    float* a = cuda_alloc(device, cells);
    float* b = cuda_alloc(device, cells);
    float* c = cuda_alloc(device, cells);
    cuda_copy_h2d(device, a, host.data(), cells);
    cuda_copy_h2d(device, b, host.data(), cells);
    cuda_gemm(device, kN, kN, kN, a, kN, b, kN, c, kN, 1.0f, 0.0f);
    cuda_sync(device);
    const auto start = std::chrono::steady_clock::now();
    constexpr int kRepeats = 4;
    for (int i = 0; i < kRepeats; ++i) {
        cuda_gemm(device, kN, kN, kN, a, kN, b, kN, c, kN, 1.0f, 0.0f);
    }
    cuda_sync(device);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    cuda_free(device, a);
    cuda_free(device, b);
    cuda_free(device, c);
    const double flops = static_cast<double>(kRepeats) * 2.0 * kN * kN * kN;
    return seconds > 0.0 ? flops / seconds / 1.0e9 : 0.0;
}

double cuda_measure_pcie_gbps(int device) {
    constexpr std::size_t kCount = 1u << 20;
    std::vector<float> host(kCount, 1.0f);
    float* device_ptr = cuda_alloc(device, kCount);
    const auto start = std::chrono::steady_clock::now();
    constexpr int kRepeats = 4;
    for (int i = 0; i < kRepeats; ++i) {
        cuda_copy_h2d(device, device_ptr, host.data(), kCount);
    }
    cuda_sync(device);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    cuda_free(device, device_ptr);
    const double bytes = static_cast<double>(kRepeats) * static_cast<double>(kCount) * sizeof(float);
    return seconds > 0.0 ? bytes / seconds / 1.0e9 : 0.0;
}

bool cuda_nccl_allreduce(float**, const int*, int, std::size_t, int) { return false; }

}  // namespace hw

std::unique_ptr<AcceleratorBackend> make_cuda_backend() {
    class CudaDriver final : public AcceleratorBackend {
    public:
        DeviceType type() const override { return DeviceType::Cuda; }
        const char* name() const override { return "cuda-cublas"; }
        int device_count() const override { return hw::cuda_device_count(); }
        std::string device_name(int index) const override { return hw::cuda_device_name(index); }
        std::uint64_t memory_bytes(int index) const override { return hw::cuda_memory_bytes(index); }
        void* allocate(int index, std::size_t bytes) override {
            const std::size_t count = (bytes + sizeof(float) - 1) / sizeof(float);
            return hw::cuda_alloc(index, count == 0 ? 1 : count);
        }
        void deallocate(int index, void* pointer) override { hw::cuda_free(index, static_cast<float*>(pointer)); }
        void copy(int dst_device, void* dst, int src_device, const void* src, std::size_t bytes) override {
            const std::size_t count = bytes / sizeof(float);
            hw::cuda_copy_p2p(dst_device, static_cast<float*>(dst), src_device, static_cast<const float*>(src), count);
            hw::cuda_sync(dst_device);
        }
        void gemm(int index, int m, int n, int k, const float* a, int lda, const float* b, int ldb, float* c, int ldc,
                  float alpha, float beta) override {
            hw::cuda_gemm(index, m, n, k, a, lda, b, ldb, c, ldc, alpha, beta);
            hw::cuda_sync(index);
        }
    };
    return std::unique_ptr<AcceleratorBackend>(new CudaDriver());
}

}  // namespace matrix_pro
