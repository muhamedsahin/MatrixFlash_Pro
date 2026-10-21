#pragma once

/**
 * @file cuda_hooks.hpp
 * @brief CUDA derlemesinde gercek, CPU derlemesinde firlatan kancalar.
 *
 * Basari durumunda bos deger donulmez. Toolkit yokken cagri MatrixProError atar.
 */

#include <cstddef>
#include <cstdint>

namespace matrix_pro {
namespace hw {

int cuda_device_count();
void cuda_gemm(int device, int m, int n, int k, const float* a, int lda, const float* b, int ldb, float* c, int ldc,
               float alpha, float beta);
void cuda_copy_h2d(int device, float* dst, const float* src, std::size_t count);
void cuda_copy_d2h(int device, float* dst, const float* src, std::size_t count);
void cuda_copy_p2p(int dst_device, float* dst, int src_device, const float* src, std::size_t count);
void cuda_add_inplace(int device, float* dst, const float* src, std::size_t count);
void cuda_max_inplace(int device, float* dst, const float* src, std::size_t count);
void cuda_scale_inplace(int device, float* data, std::size_t count, float scale);
float* cuda_alloc(int device, std::size_t count);
void cuda_free(int device, float* pointer);
void cuda_sync(int device);
bool cuda_peer_access(int a, int b);
bool cuda_nvlink(int a, int b);
void cuda_enable_peer(int a, int b);
const char* cuda_device_name(int device);
std::uint64_t cuda_memory_bytes(int device);
double cuda_measure_gflops(int device);
double cuda_measure_pcie_gbps(int device);
bool cuda_nccl_allreduce(float** ptrs, const int* devices, int rank_count, std::size_t count, int reduce_op);
void cuda_allreduce(float** data, const int* devices, int rank_count, std::size_t count, int reduce_op, int algorithm);

}  // namespace hw
}  // namespace matrix_pro
