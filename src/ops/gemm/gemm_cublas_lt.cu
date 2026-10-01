// Default large-path entry used by gemm_rowmajor().
// Delegates to the per-thread plan cache. The first call for a shape builds
// descriptors and asks cuBLASLt for a heuristic; later calls replay that plan.
// No bias and no fused epilogue on this entry — fused GEMM goes through
// gemm_bias_epilogue() in gemm_fused.cu.

#include "matrix_pro/detail/gemm/gemm_api.hpp"

namespace matrix_pro { namespace detail { namespace gemm {
void gemm_cublas_lt(const float* a, const float* b, float* c,
                    int m, int n, int k, cudaStream_t stream) {
    gemm_cached_raw(a, b, c, nullptr, m, n, k, Epilogue::none, stream);
}
} } }
