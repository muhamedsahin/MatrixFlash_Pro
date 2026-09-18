#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

#ifdef MATRIX_PRO_HAS_CUDNN
#include <cudnn.h>
#endif

#include <cstddef>

#include "matrix_pro/core/errors.hpp"

// cuSOLVER handle forward declaration (opaque context pointer; the full type is
// provided by <cusolverDn.h> in translation units that instantiate it).
struct cusolverDnContext;

namespace matrix_pro {

void* allocate_device_memory(std::size_t bytes);
void free_device_memory(void* pointer);
// Zeroes a device buffer. Required by kernels that accumulate with atomicAdd
// into freshly allocated (and therefore uninitialized) memory.
void zero_device_memory(void* pointer, std::size_t bytes);
void checkCuda(cudaError_t status, const char* operation);
void synchronize();
cudaStream_t compute_stream();
cublasHandle_t& cublas_handle();
// Number of SMs on the current device (cached). Used to size persistent-style
// reduction grids so partial buffers stay small and host reduction is cheap.
int sm_count();

// --- device management -----------------------------------------------------
// The library keeps one execution context (CUDA stream + cuBLAS/cuSOLVER/cuDNN
// handles) per (host thread, CUDA device) pair. Separate host threads therefore
// never share a stream and independent devices get their own handles, so
// concurrently executing operations from different threads are safe. The device
// memory pool is shared but internally synchronized and tagged per device.
int device_count();
int current_device();
// Fails with InvalidArgumentError when `device` is out of range.
void set_device(int device);
// Blocks until every stream owned by the current thread has finished. Call
// synchronize() for the current device only, or this for all touched devices.
void synchronize_all_devices();

// RAII helper restoring the previously selected device on scope exit. The
// destructor never throws; use set_device() when you need error reporting.
class DeviceScope {
public:
    explicit DeviceScope(int device);
    ~DeviceScope();
    DeviceScope(const DeviceScope&) = delete;
    DeviceScope& operator=(const DeviceScope&) = delete;
    int previous_device() const noexcept { return previous_device_; }

private:
    int previous_device_ = 0;
};

// cuSOLVER handle for the current thread and device (lazily created, reused).
cusolverDnContext*& cusolver_handle();
#ifdef MATRIX_PRO_HAS_CUDNN
cudnnHandle_t& cudnn_handle();
#endif

}
