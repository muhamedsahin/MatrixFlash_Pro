#include "matrix_pro/detail/gemm/gemm_api.hpp"
#include "matrix_pro/detail/gemm/gemm_config.cuh"
#include "matrix_pro/core/cuda_utils.hpp"

// Shape-aware GEMM front door for the legacy row-major path
// (Matrix::operator*, multiply, multiply_into).
//
// Layout contract: A is m×k, B is k×n, C is m×n, all contiguous row-major.
// The public GemmPlan API (ops/gemm.hpp, gemm_plan.cu) is a separate reusable
// plan with explicit FP32/TF32 and transpose/epilogue options. This function
// keeps the historical default policy encoded in select_backend():
//
//   * k == 0 writes a zero C and returns. Negative/empty M or N is a no-op.
//   * m == 1 and k >= 1024 goes to cublasLt. The long reduction was slower in
//     the custom GEMV kernel; vendor math here stays strict FP32 (see
//     gemm_cached_raw) so the speedup is not bought by dropping mantissa bits.
//   * n <= 1, m <= 1, or a skinny N with a long K uses the warp GEMV kernel.
//   * every dimension <= 64 uses the register micro-kernel. cuBLAS launch and
//     heuristic cost dominates the arithmetic at that size.
//   * everything else uses the algo-cached cublasLt path (TF32 on the default
//     raw policy). GemmEx remains the fallback inside the plan if Lt fails.
//
// The tiled shared-memory kernel is still compiled (gemm_tiled.cu) but
// select_backend() currently prefers Lt once the problem leaves the micro/GEMV
// window. Thresholds live in detail/gemm/gemm_config.cuh.

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
