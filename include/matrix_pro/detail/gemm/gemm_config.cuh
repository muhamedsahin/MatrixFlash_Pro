#pragma once

// Shape thresholds and tile geometry for the shape-aware GEMM dispatcher.
// Tuned for Ampere (sm_80/sm_86) with TF32 tensor-op math via cuBLAS/cublasLt;
// custom kernels cover the regimes where vendor libraries pay launch / heuristic
// overhead that dominates arithmetic.

#include <cstddef>

namespace matrix_pro {
namespace detail {
namespace gemm {

// Below this product of dimensions, a register-tiled micro-kernel beats cuBLAS
// (handle / heuristic / workspace setup dominates).
constexpr int kSmallMaxDim = 64;
constexpr long long kSmallMaxFlops = 64ll * 64 * 64 * 2;

// Matrix-vector (or very skinny) problems: custom warp GEMV.
constexpr int kGemvMaxInner = 1;
constexpr int kSkinnyMaxN = 4;

// Prefer cublasLt (algo search + TF32) above this flops estimate.
constexpr long long kLtMinFlops = 256ll * 256 * 256 * 2;

// Persistent cublasLt workspace (shared with the execution context when possible).
constexpr std::size_t kLtWorkspaceBytes = 64u * 1024u * 1024u;

// Tiled SGEMM block geometry (row-major A/B/C, float4 vector loads).
constexpr int kTileM = 128;
constexpr int kTileN = 128;
constexpr int kTileK = 8;
constexpr int kThreadTileM = 8;
constexpr int kThreadTileN = 8;

enum class GemmBackend {
    micro,   // register tiled, tiny shapes
    gemv,    // matrix × vector / skinny
    tiled,   // custom shared-memory SGEMM
    cublas,  // cublasGemmEx fallback
    cublaslt // preferred large-path
};

inline GemmBackend select_backend(int m, int n, int k) noexcept {
    if (m <= 0 || n <= 0 || k <= 0) return GemmBackend::cublas;
    if (n <= kGemvMaxInner || m <= kGemvMaxInner) return GemmBackend::gemv;
    if (n <= kSkinnyMaxN && k >= 128) return GemmBackend::gemv;
    const long long flops =
        2ll * static_cast<long long>(m) * static_cast<long long>(n) *
        static_cast<long long>(k);
    if (m <= kSmallMaxDim && n <= kSmallMaxDim && k <= kSmallMaxDim &&
        flops <= kSmallMaxFlops) {
        return GemmBackend::micro;
    }
    // Large / medium: prefer algo-cached cublasLt (TF32). After the first call
    // per shape the heuristic is reused, which keeps us on the vendor peak
    // path and often matches or beats a cold GemmEx pick. GemmEx TENSOR_OP
    // remains the fallback inside gemm_cublas_lt / for smaller medium shapes.
    if (flops >= kLtMinFlops || (m >= 256 && n >= 256 && k >= 256)) {
        return GemmBackend::cublaslt;
    }
    return GemmBackend::cublas;
}

} // namespace gemm
} // namespace detail
} // namespace matrix_pro
