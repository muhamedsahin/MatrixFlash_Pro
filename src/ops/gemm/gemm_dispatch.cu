#include "matrix_pro/detail/gemm/gemm_api.hpp"
#include "matrix_pro/detail/gemm/gemm_config.cuh"
#include "matrix_pro/core/cuda_utils.hpp"

namespace matrix_pro {
namespace detail {
namespace gemm {

void gemm_rowmajor(const float* a, const float* b, float* c,
                   int m, int n, int k, cudaStream_t stream) {
    if(m<=0 || n<=0) return;
    if(k==0) {
        checkCuda(cudaMemsetAsync(c,0,static_cast<std::size_t>(m)*n*sizeof(float),stream),"empty GEMM inner dimension");
        return;
    }
    switch (select_backend(m, n, k)) {
        case GemmBackend::micro:
            gemm_micro(a, b, c, m, n, k, stream);
            break;
        case GemmBackend::gemv:
            gemm_gemv(a, b, c, m, n, k, stream);
            break;
        case GemmBackend::tiled:
            gemm_tiled(a, b, c, m, n, k, stream);
            break;
        case GemmBackend::cublaslt:
            gemm_cublas_lt(a, b, c, m, n, k, stream);
            break;
        case GemmBackend::cublas:
        default:
            gemm_cublas_ex(a, b, c, m, n, k, stream);
            break;
    }
}

} // namespace gemm
} // namespace detail
} // namespace matrix_pro
