#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

#ifdef MATRIX_PRO_HAS_CUDNN
#include <cudnn.h>
#endif

#include <cstddef>

// cuSOLVER handle forward declaration (opaque context pointer; the full type is
// provided by <cusolverDn.h> in translation units that instantiate it).
struct cusolverDnContext;

namespace matrix_pro {

void* allocate_device_memory(std::size_t bytes);
void free_device_memory(void* pointer);
void checkCuda(cudaError_t status, const char* operation);
void synchronize();
cudaStream_t compute_stream();
cublasHandle_t& cublas_handle();

// Shared cuSOLVER handle (lazily created once, reused process-wide). All linear
// algebra routines draw from a single handle to avoid repeated creation cost.
cusolverDnContext*& cusolver_handle();
#ifdef MATRIX_PRO_HAS_CUDNN
cudnnHandle_t& cudnn_handle();
#endif

}
