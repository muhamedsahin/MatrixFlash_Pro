// Column-bias epilogue used when a fused cuBLASLt epilogue is unavailable,
// and the public fused GEMM entry.
//
// epilogue_col_bias_kernel walks C in row-major order and adds bias[col].
// mode 0 is bias only, mode 1 is ReLU(bias), mode 2 is GELU(bias).
// GELU here is the tanh approximation, matching the nn::gelu used elsewhere.
//
// gemm_bias_epilogue() prefers one Lt matmul whose epilogue applies the bias
// and the activation inside the vendor kernel (no second global-memory pass).
// If that plan cannot run, gemm_cached_raw falls back to a plain GEMM plus
// apply_epilogue(). This file does not claim a measured speedup factor: the
// 1 Oct 2026 suite did not publish a separate fused-vs-chain table.

#include "matrix_pro/detail/gemm/gemm_api.hpp"
#include "matrix_pro/core/cuda_utils.hpp"


namespace matrix_pro {
namespace detail {
namespace gemm {
namespace {

__global__ void epilogue_col_bias_kernel(float* __restrict__ c,
                                         const float* __restrict__ bias,
                                         int m, int n, int mode) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = static_cast<std::size_t>(m) * n;
    if (index >= total) return;
    const int col = static_cast<int>(index % static_cast<std::size_t>(n));
    float v = c[index] + bias[col];
    if (mode == 1) {
        v = v > 0.0f ? v : 0.0f;
    } else if (mode == 2) {
        const float x3 = v * v * v;
        v = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * x3)));
    }
    c[index] = v;
}

} // namespace

void apply_epilogue(float* c, const float* bias, int m, int n, Epilogue epi,
                    cudaStream_t stream) {
    if (epi == Epilogue::none || bias == nullptr || m <= 0 || n <= 0) return;
    int mode = 0;
    if (epi == Epilogue::bias_relu) mode = 1;
    else if (epi == Epilogue::bias_gelu) mode = 2;
    const std::size_t total = static_cast<std::size_t>(m) * n;
    const unsigned block = 256;
    const unsigned grid =
        static_cast<unsigned>((total + block - 1) / block);
    epilogue_col_bias_kernel<<<grid, block, 0, stream>>>(c, bias, m, n, mode);
    checkCuda(cudaGetLastError(), "epilogue kernel launch");
}

void gemm_bias_epilogue(const float* a, const float* b, float* c,
                        const float* bias, int m, int n, int k, Epilogue epi,
                        cudaStream_t stream) {
    gemm_cached_raw(a, b, c, bias, m, n, k, epi, stream);
}

} // namespace gemm
} // namespace detail
} // namespace matrix_pro
