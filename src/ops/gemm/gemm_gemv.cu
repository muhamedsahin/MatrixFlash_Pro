#include "matrix_pro/detail/gemm/gemm_api.hpp"
#include "matrix_pro/core/cuda_utils.hpp"

namespace matrix_pro {
namespace detail {
namespace gemm {
namespace {

// Warp-cooperative GEMV: y[m] = A[m,k] * x[k]. Each warp owns one row.
__global__ void gemv_rows_kernel(const float* __restrict__ a,
                                 const float* __restrict__ x,
                                 float* __restrict__ y,
                                 int m, int k) {
    const int row = static_cast<int>(blockIdx.x * blockDim.y + threadIdx.y);
    if (row >= m) return;

    const float* row_ptr = a + static_cast<std::size_t>(row) * k;
    float sum = 0.0f;
    for (int col = static_cast<int>(threadIdx.x); col < k; col += static_cast<int>(blockDim.x)) {
        sum += row_ptr[col] * x[col];
    }
    // Warp reduce
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (threadIdx.x == 0) y[row] = sum;
}

// Skinny N (2..4): each warp computes one row × N columns.
__global__ void gemm_skinny_n_kernel(const float* __restrict__ a,
                                     const float* __restrict__ b,
                                     float* __restrict__ c,
                                     int m, int n, int k) {
    const int row = static_cast<int>(blockIdx.x * blockDim.y + threadIdx.y);
    if (row >= m) return;

    float acc[4] = {};
    const float* row_ptr = a + static_cast<std::size_t>(row) * k;
    for (int p = static_cast<int>(threadIdx.x); p < k; p += static_cast<int>(blockDim.x)) {
        const float av = row_ptr[p];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (j < n) acc[j] += av * b[static_cast<std::size_t>(p) * n + j];
        }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc[j] += __shfl_down_sync(0xffffffffu, acc[j], offset);
        }
    }
    if (threadIdx.x == 0) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (j < n) c[static_cast<std::size_t>(row) * n + j] = acc[j];
        }
    }
}

// Vector × matrix: y[n] = x[k]^T * B[k,n]  (m == 1).
__global__ void gemv_cols_kernel(const float* __restrict__ x,
                                 const float* __restrict__ b,
                                 float* __restrict__ y,
                                 int n, int k) {
    // Eight warps cooperate along K while lanes load consecutive B columns.
    // This exposes K parallelism and launches one block per 32 columns instead
    // of leaving almost every SM idle behind a serial K loop.
    __shared__ float partial[8][32];
    const int col = static_cast<int>(blockIdx.x * 32 + threadIdx.x);
    float sum = 0.0f;
    if(col<n) for (int p = threadIdx.y; p < k; p += 8) {
        sum = fmaf(x[p], b[static_cast<std::size_t>(p) * n + col], sum);
    }
    partial[threadIdx.y][threadIdx.x]=sum;
    __syncthreads();
    if(threadIdx.y==0 && col<n) {
        float total=0;
        #pragma unroll
        for(int i=0;i<8;++i)total+=partial[i][threadIdx.x];
        y[col]=total;
    }
}

} // namespace

void gemm_gemv(const float* a, const float* b, float* c,
               int m, int n, int k, cudaStream_t stream) {
    if (n == 1) {
        dim3 block(32, 4);
        const unsigned rows_per_block = block.y;
        const unsigned grid =
            (static_cast<unsigned>(m) + rows_per_block - 1) / rows_per_block;
        gemv_rows_kernel<<<grid, block, 0, stream>>>(a, b, c, m, k);
    } else if (m == 1) {
        dim3 block(32, 8);
        const unsigned grid = (static_cast<unsigned>(n) + 31) / 32;
        gemv_cols_kernel<<<grid, block, 0, stream>>>(a, b, c, n, k);
    } else if (n <= 4) {
        dim3 block(32, 4);
        const unsigned grid =
            (static_cast<unsigned>(m) + block.y - 1) / block.y;
        gemm_skinny_n_kernel<<<grid, block, 0, stream>>>(a, b, c, m, n, k);
    } else {
        // Fallback: should not be selected, but stay correct.
        gemm_cublas_ex(a, b, c, m, n, k, stream);
        return;
    }
    checkCuda(cudaGetLastError(), "gemv kernel launch");
}

} // namespace gemm
} // namespace detail
} // namespace matrix_pro
