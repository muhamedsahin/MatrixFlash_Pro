#include "matrix_pro/detail/gemm/gemm_api.hpp"
#include "matrix_pro/detail/gemm/gemm_config.cuh"
#include "matrix_pro/core/cuda_utils.hpp"

#include <cublasLt.h>
#include <cublas_v2.h>

#include <cstdint>
#include <mutex>
#include <unordered_map>

namespace matrix_pro {
namespace detail {
namespace gemm {
namespace {

struct ShapeKey {
    int m = 0;
    int n = 0;
    int k = 0;
    bool operator==(const ShapeKey& o) const noexcept {
        return m == o.m && n == o.n && k == o.k;
    }
};

struct ShapeKeyHash {
    std::size_t operator()(const ShapeKey& s) const noexcept {
        // 64-bit mix of three positive ints.
        std::uint64_t h = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(s.m)) << 42) ^
                          (static_cast<std::uint64_t>(static_cast<std::uint32_t>(s.n)) << 21) ^
                          static_cast<std::uint32_t>(s.k);
        h ^= h >> 33;
        h *= 0xff51afd7ed558ccdULL;
        h ^= h >> 33;
        return static_cast<std::size_t>(h);
    }
};

struct CachedAlgo {
    cublasLtMatmulAlgo_t algo{};
    bool valid = false;
};

struct LtState {
    cublasLtHandle_t handle = nullptr;
    void* workspace = nullptr;
    std::size_t workspace_bytes = kLtWorkspaceBytes;
    std::mutex mutex;
    std::unordered_map<ShapeKey, CachedAlgo, ShapeKeyHash> algo_cache;

    LtState() {
        if (cublasLtCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
            handle = nullptr;
            return;
        }
        if (cudaMalloc(&workspace, workspace_bytes) != cudaSuccess) {
            cudaGetLastError();
            workspace = nullptr;
            workspace_bytes = 0;
        }
    }

    ~LtState() {
        if (workspace != nullptr) cudaFree(workspace);
        if (handle != nullptr) cublasLtDestroy(handle);
    }

    static LtState& instance() {
        static LtState state;
        return state;
    }
};

void destroy_layouts(cublasLtMatmulDesc_t op, cublasLtMatrixLayout_t a,
                     cublasLtMatrixLayout_t b, cublasLtMatrixLayout_t c,
                     cublasLtMatmulPreference_t pref) {
    if (pref) cublasLtMatmulPreferenceDestroy(pref);
    if (c) cublasLtMatrixLayoutDestroy(c);
    if (b) cublasLtMatrixLayoutDestroy(b);
    if (a) cublasLtMatrixLayoutDestroy(a);
    if (op) cublasLtMatmulDescDestroy(op);
}

bool run_lt(LtState& lt, const float* a, const float* b, float* c,
            int m, int n, int k, cudaStream_t stream) {
    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatrixLayout_t adesc = nullptr;
    cublasLtMatrixLayout_t bdesc = nullptr;
    cublasLtMatrixLayout_t cdesc = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;

    if (cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F_FAST_TF32,
                                 CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) {
        return false;
    }
    if (cublasLtMatrixLayoutCreate(&adesc, CUDA_R_32F, n, k, n) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutCreate(&bdesc, CUDA_R_32F, k, m, k) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutCreate(&cdesc, CUDA_R_32F, n, m, n) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulPreferenceCreate(&pref) != CUBLAS_STATUS_SUCCESS) {
        destroy_layouts(op_desc, adesc, bdesc, cdesc, pref);
        return false;
    }
    cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &lt.workspace_bytes,
        sizeof(lt.workspace_bytes));

    const ShapeKey key{m, n, k};
    CachedAlgo& cached = lt.algo_cache[key];

    // First sighting of this shape: ask for several heuristics and keep the
    // top-ranked algo so later calls skip the search and stay on the peak path.
    if (!cached.valid) {
        cublasLtMatmulHeuristicResult_t results[8]{};
        int returned = 0;
        if (cublasLtMatmulAlgoGetHeuristic(lt.handle, op_desc, adesc, bdesc,
                                           cdesc, cdesc, pref, 8, results,
                                           &returned) == CUBLAS_STATUS_SUCCESS &&
            returned > 0) {
            cached.algo = results[0].algo;
            cached.valid = true;
        }
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const cublasLtMatmulAlgo_t* algo_ptr = cached.valid ? &cached.algo : nullptr;
    const cublasStatus_t status = cublasLtMatmul(
        lt.handle, op_desc, &alpha, b, adesc, a, bdesc, &beta, c, cdesc, c,
        cdesc, algo_ptr, lt.workspace, lt.workspace_bytes, stream);

    destroy_layouts(op_desc, adesc, bdesc, cdesc, pref);
    return status == CUBLAS_STATUS_SUCCESS;
}

} // namespace

void gemm_cublas_lt(const float* a, const float* b, float* c,
                    int m, int n, int k, cudaStream_t stream) {
    LtState& lt = LtState::instance();
    if (lt.handle == nullptr) {
        gemm_cublas_ex(a, b, c, m, n, k, stream);
        return;
    }

    std::lock_guard<std::mutex> lock(lt.mutex);
    if (!run_lt(lt, a, b, c, m, n, k, stream)) {
        // Invalidate a bad cached algo so the next call re-queries.
        lt.algo_cache.erase(ShapeKey{m, n, k});
        gemm_cublas_ex(a, b, c, m, n, k, stream);
    }
}

} // namespace gemm
} // namespace detail
} // namespace matrix_pro
