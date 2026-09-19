#include "matrix_pro/detail/gemm/gemm_api.hpp"
#include "matrix_pro/core/cuda_utils.hpp"

#include <cstdint>

namespace matrix_pro {
namespace detail {
namespace gemm {
namespace {

// Register-tiled micro GEMM for tiny shapes (launch + heuristic overhead of
// cuBLAS dominates here). Each thread owns one C element; K is unrolled via
// float4 when aligned.
__global__ void micro_gemm_kernel(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ c,
                                  int m, int n, int k) {
    const int col = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int row = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (row >= m || col >= n) return;

    float acc = 0.0f;
    const float* a_row = a + static_cast<std::size_t>(row) * k;
    int p = 0;
    // Vectorized K when B columns are contiguous and K is multiple of 4.
    if ((n % 4) == 0 && (k % 4) == 0 &&
        (reinterpret_cast<uintptr_t>(b) & 0xFu) == 0u &&
        (reinterpret_cast<uintptr_t>(a_row) & 0xFu) == 0u) {
        for (; p + 3 < k; p += 4) {
            const float4 av = *reinterpret_cast<const float4*>(a_row + p);
            acc += av.x * b[static_cast<std::size_t>(p) * n + col];
            acc += av.y * b[static_cast<std::size_t>(p + 1) * n + col];
            acc += av.z * b[static_cast<std::size_t>(p + 2) * n + col];
            acc += av.w * b[static_cast<std::size_t>(p + 3) * n + col];
        }
    }
    for (; p < k; ++p) {
        acc += a_row[p] * b[static_cast<std::size_t>(p) * n + col];
    }
    c[static_cast<std::size_t>(row) * n + col] = acc;
}

// 2D register blocking: each thread computes a 4x4 tile of C. Much better ILP
// on Ampere for 32..64 sized problems.
__global__ void micro_gemm_4x4_kernel(const float* __restrict__ a,
                                      const float* __restrict__ b,
                                      float* __restrict__ c,
                                      int m, int n, int k) {
    const int col0 = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) * 4);
    const int row0 = static_cast<int>((blockIdx.y * blockDim.y + threadIdx.y) * 4);
    if (row0 >= m || col0 >= n) return;

    float acc[4][4] = {};
    for (int p = 0; p < k; ++p) {
        float av[4];
        float bv[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            av[i] = (row0 + i < m) ? a[static_cast<std::size_t>(row0 + i) * k + p]
                                   : 0.0f;
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            bv[j] = (col0 + j < n) ? b[static_cast<std::size_t>(p) * n + (col0 + j)]
                                   : 0.0f;
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                acc[i][j] += av[i] * bv[j];
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        if (row0 + i >= m) continue;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (col0 + j < n) {
                c[static_cast<std::size_t>(row0 + i) * n + (col0 + j)] = acc[i][j];
            }
        }
    }
}

} // namespace

void gemm_micro(const float* a, const float* b, float* c,
                int m, int n, int k, cudaStream_t stream) {
    if (m >= 16 && n >= 16 && (m % 4 == 0) && (n % 4 == 0)) {
        dim3 block(8, 8);
        dim3 grid((static_cast<unsigned>(n) / 4 + block.x - 1) / block.x,
                  (static_cast<unsigned>(m) / 4 + block.y - 1) / block.y);
        micro_gemm_4x4_kernel<<<grid, block, 0, stream>>>(a, b, c, m, n, k);
    } else {
        dim3 block(16, 16);
        dim3 grid((static_cast<unsigned>(n) + 15) / 16,
                  (static_cast<unsigned>(m) + 15) / 16);
        micro_gemm_kernel<<<grid, block, 0, stream>>>(a, b, c, m, n, k);
    }
    checkCuda(cudaGetLastError(), "micro gemm kernel launch");
}

} // namespace gemm
} // namespace detail
} // namespace matrix_pro
