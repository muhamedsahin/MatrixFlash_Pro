// Part 1: includes + helpers. cublasLt + raw cuSOLVER + Thrust/CUB live here
// (all ship with the CUDA toolkit: zero extra installs for FAZ A).
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "benchmark_support.hpp"
#include "matrix_pro/core/cuda_utils.hpp"
#include "matrix_pro/core/matrix.hpp"
#include "matrix_pro/ops/linalg.hpp"
#include "matrix_pro/ops/product.hpp"
#include "matrix_pro/ops/reductions.hpp"

#if defined(__has_include)
#if __has_include(<thrust/reduce.h>)
#define MATRIX_PRO_EXTERNAL_HAS_THRUST 1
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#endif
#if __has_include(<cub/cub.cuh>)
#define MATRIX_PRO_EXTERNAL_HAS_CUB 1
#include <cub/cub.cuh>
#endif
#endif

namespace matrix_pro {
namespace bench {
namespace {
// Owns a cublasLt handle + workspace for the whole run.
struct LtContext {
    cublasLtHandle_t handle = nullptr;
    void* workspace = nullptr;
    std::size_t workspace_bytes = 32u * 1024u * 1024u;
    LtContext() {
        if (cublasLtCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cublasLtCreate failed");
        }
        if (cudaMalloc(&workspace, workspace_bytes) != cudaSuccess) {
            cudaGetLastError();
            workspace = nullptr;
            workspace_bytes = 0;
        }
    }
    ~LtContext() {
        if (workspace != nullptr) cudaFree(workspace);
        if (handle != nullptr) cublasLtDestroy(handle);
    }
};

void check_lt(cublasStatus_t status, const char* op) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(op) + ": cublasLt status " +
                                 std::to_string(static_cast<int>(status)));
    }
}

// One timed cublasLtMatmul on pre-allocated row-major buffers.
// Layout trick: row-major (m x k) * (k x n) == column-major (n x m) * (m x k)
// with the operands swapped — the same trick matrix_pro::multiply uses.
void lt_gemm_once(LtContext& ctx, const float* d_left, const float* d_right,
                  float* d_out, int m, int n, int k) {
    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatrixLayout_t adesc = nullptr, bdesc = nullptr, cdesc = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    check_lt(cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F),
             "lt desc create");
    check_lt(cublasLtMatrixLayoutCreate(&adesc, CUDA_R_32F, n, k, n),
             "lt adesc");
    check_lt(cublasLtMatrixLayoutCreate(&bdesc, CUDA_R_32F, k, m, k),
             "lt bdesc");
    check_lt(cublasLtMatrixLayoutCreate(&cdesc, CUDA_R_32F, n, m, n),
             "lt cdesc");
    check_lt(cublasLtMatmulPreferenceCreate(&pref), "lt pref");
    check_lt(cublasLtMatmulPreferenceSetAttribute(
                 pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                 &ctx.workspace_bytes, sizeof(ctx.workspace_bytes)),
             "lt workspace attr");
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const cublasStatus_t status = cublasLtMatmul(
        ctx.handle, op_desc, &alpha, d_right, adesc, d_left, bdesc, &beta,
        d_out, cdesc, d_out, cdesc, nullptr, ctx.workspace,
        ctx.workspace_bytes, compute_stream());
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(cdesc);
    cublasLtMatrixLayoutDestroy(bdesc);
    cublasLtMatrixLayoutDestroy(adesc);
    cublasLtMatmulDescDestroy(op_desc);
    check_lt(status, "cublasLtMatmul");
}

void run_gemm_lt(const Options& options, std::vector<Result>& out,
                 const std::vector<std::size_t>& sizes) {
    LtContext ctx;
    for (const std::size_t n : sizes) {
        if (n < 32) {
            continue;
        }
        if (!memory_available(4 * n * n * sizeof(float))) {
            if (!options.quiet) {
                std::cout << "  (skipping external-gemm " << n << "x" << n
                          << ": not enough free device memory)\n";
            }
            continue;
        }
        const std::string label = format_matrix_label(n, n);
        const double flops = 2.0 * static_cast<double>(n) * n * n;
        float* d_left = nullptr;
        float* d_right = nullptr;
        float* d_out = nullptr;
        checkCuda(cudaMalloc(&d_left, n * n * sizeof(float)), "lt cudaMalloc");
        checkCuda(cudaMalloc(&d_right, n * n * sizeof(float)), "lt cudaMalloc");
        checkCuda(cudaMalloc(&d_out, n * n * sizeof(float)), "lt cudaMalloc");
        checkCuda(cudaMemset(d_left, 0, n * n * sizeof(float)), "lt memset");
        checkCuda(cudaMemset(d_right, 0, n * n * sizeof(float)), "lt memset");
        const int side = static_cast<int>(n);
        const Stats ms = sample_ms(
            [&] { lt_gemm_once(ctx, d_left, d_right, d_out, side, side, side); },
            options);
        cudaFree(d_left);
        cudaFree(d_right);
        cudaFree(d_out);
        add_result(out, "external-gemm", label, n, ms,
                   to_gflops(flops, ms.median), "GFLOPS",
                   "cublasLt pre-alloc, no wrapper (measured)");
    }
}

void run_external_gemm(const Options& options, std::vector<Result>& out) {
    const std::vector<std::size_t> sizes =
        options.sizes.empty() ? std::vector<std::size_t>{256, 512, 1024, 2048}
                              : options.sizes;
    run_gemm_lt(options, out, sizes);
}

} // namespace

const BenchmarkInfo& external_gemm_benchmark() {
    static const BenchmarkInfo info{
        "external_gemm",
        "ceiling GEMM: raw cublasLtMatmul (pre-allocated, no wrapper)",
        run_external_gemm};
    return info;
}

} // namespace bench
} // namespace matrix_pro
