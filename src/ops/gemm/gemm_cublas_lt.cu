#include "matrix_pro/detail/gemm/gemm_api.hpp"

namespace matrix_pro { namespace detail { namespace gemm {
void gemm_cublas_lt(const float* a, const float* b, float* c,
                    int m, int n, int k, cudaStream_t stream) {
    gemm_cached_raw(a, b, c, nullptr, m, n, k, Epilogue::none, stream);
}
} } }
