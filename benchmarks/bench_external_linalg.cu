// Part 1: includes. Raw cuSOLVER (getrf/getrs/geqrf/gesvd/syevj) + Thrust/CUB.
// All ship with the CUDA toolkit: zero extra installs for FAZ A.
#include <cmath>
#include <cstddef>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "benchmark_support.hpp"
#include "matrix_pro/core/cuda_utils.hpp"
#include "matrix_pro/core/matrix.hpp"
#include "matrix_pro/ops/linalg.hpp"
#include "matrix_pro/ops/reductions.hpp"

#if defined(__has_include)
#if __has_include(<thrust/reduce.h>)
#define MATRIX_PRO_EXTERNAL_HAS_THRUST 1
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
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

void check_solver(cusolverStatus_t status, const char* op) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(op) + ": cusolver status " +
                                 std::to_string(static_cast<int>(status)));
    }
}

// Column-major SPD host matrix: n*I + 0.5 (matches Matrix::solve test data).
std::vector<float> host_spd_cm(std::size_t n) {
    std::vector<float> host(n * n, 0.5f);
    for (std::size_t i = 0; i < n; ++i) host[i * n + i] += static_cast<float>(n);
    return host;
}

// Column-major deterministic general matrix (matches bench_linalg data).
std::vector<float> host_general_cm(std::size_t n) {
    std::vector<float> host(n * n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            const std::size_t idx = i * n + j;
            host[j * n + i] =
                static_cast<float>((idx * 2654435761u) % 10000u) / 1000.0f + 0.1f;
        }
    return host;
}

float* upload_cm(const std::vector<float>& host) {
    float* dev = nullptr;
    checkCuda(cudaMalloc(&dev, host.size() * sizeof(float)), "ext cudaMalloc");
    checkCuda(cudaMemcpy(dev, host.data(), host.size() * sizeof(float),
                         cudaMemcpyHostToDevice),
              "ext upload");
    return dev;
}

// Raw LU solve: getrf + getrs on column-major buffers (no row/col transpose
// kernels, no Matrix alloc). Timed region = factorize + triangular solve.
void raw_solve_once(std::size_t n) {
    const int side = static_cast<int>(n);
    float* d_a = upload_cm(host_spd_cm(n));
    float* d_b = upload_cm(host_spd_cm(n));
    int* d_ipiv = nullptr;
    int* d_info = nullptr;
    float* d_work = nullptr;
    checkCuda(cudaMalloc(&d_ipiv, n * sizeof(int)), "ext ipiv");
    checkCuda(cudaMalloc(&d_info, sizeof(int)), "ext info");
    int lwork = 0;
    check_solver(cusolverDnSgetrf_bufferSize(cusolver_handle(), side, side, d_a,
                                             side, &lwork),
                 "raw getrf bufsize");
    checkCuda(cudaMalloc(&d_work, static_cast<std::size_t>(lwork) * sizeof(float)),
              "ext work");
    check_solver(cusolverDnSgetrf(cusolver_handle(), side, side, d_a, side,
                                  d_work, d_ipiv, d_info),
                 "raw getrf");
    check_solver(cusolverDnSgetrs(cusolver_handle(), CUBLAS_OP_N, side, side,
                                  d_a, side, d_ipiv, d_b, side, d_info),
                 "raw getrs");
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_ipiv);
    cudaFree(d_info);
    cudaFree(d_work);
}

// Raw QR: geqrf only (no orgqr, no R extraction). Reports factorize cost.
void raw_qr_once(std::size_t n) {
    const int side = static_cast<int>(n);
    float* d_a = upload_cm(host_general_cm(n));
    float* d_tau = nullptr;
    int* d_info = nullptr;
    float* d_work = nullptr;
    checkCuda(cudaMalloc(&d_tau, n * sizeof(float)), "ext tau");
    checkCuda(cudaMalloc(&d_info, sizeof(int)), "ext info");
    int lwork = 0;
    check_solver(cusolverDnSgeqrf_bufferSize(cusolver_handle(), side, side, d_a,
                                             side, &lwork),
                 "raw geqrf bufsize");
    checkCuda(cudaMalloc(&d_work, static_cast<std::size_t>(lwork) * sizeof(float)),
              "ext work");
    check_solver(cusolverDnSgeqrf(cusolver_handle(), side, side, d_a, side,
                                  d_tau, d_work, lwork, d_info),
                 "raw geqrf");
    cudaFree(d_a);
    cudaFree(d_tau);
    cudaFree(d_info);
    cudaFree(d_work);
}

// Raw SVD: gesvdj (Jacobi) — CUDA 13 requires an explicit workspace.
void raw_svd_once(std::size_t n) {
    const int side = static_cast<int>(n);
    float* d_a = upload_cm(host_general_cm(n));
    float* d_s = nullptr;
    float* d_u = nullptr;
    float* d_v = nullptr;
    float* d_work = nullptr;
    int* d_info = nullptr;
    checkCuda(cudaMalloc(&d_s, n * sizeof(float)), "ext s");
    checkCuda(cudaMalloc(&d_u, n * n * sizeof(float)), "ext u");
    checkCuda(cudaMalloc(&d_v, n * n * sizeof(float)), "ext v");
    checkCuda(cudaMalloc(&d_info, sizeof(int)), "ext info");
    gesvdjInfo_t params = nullptr;
    check_solver(cusolverDnCreateGesvdjInfo(&params), "raw gesvdj info");
    int lwork = 0;
    check_solver(cusolverDnSgesvdj_bufferSize(
                     cusolver_handle(), CUSOLVER_EIG_MODE_VECTOR, 1, side, side,
                     d_a, side, d_s, d_u, side, d_v, side, &lwork, params),
                 "raw gesvdj bufsize");
    checkCuda(cudaMalloc(&d_work, static_cast<std::size_t>(lwork) * sizeof(float)),
              "ext svd work");
    check_solver(cusolverDnSgesvdj(cusolver_handle(), CUSOLVER_EIG_MODE_VECTOR, 1,
                                   side, side, d_a, side, d_s, d_u, side, d_v,
                                   side, d_work, lwork, d_info, params),
                 "raw gesvdj");
    cusolverDnDestroyGesvdjInfo(params);
    cudaFree(d_a);
    cudaFree(d_s);
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_work);
    cudaFree(d_info);
}

// Raw symmetric eig: syevj (Jacobi) — CUDA 13 requires an explicit workspace.
void raw_eigen_once(std::size_t n) {
    const int side = static_cast<int>(n);
    float* d_a = upload_cm(host_spd_cm(n));
    float* d_w = nullptr;
    float* d_work = nullptr;
    int* d_info = nullptr;
    checkCuda(cudaMalloc(&d_w, n * sizeof(float)), "ext w");
    checkCuda(cudaMalloc(&d_info, sizeof(int)), "ext info");
    syevjInfo_t params = nullptr;
    check_solver(cusolverDnCreateSyevjInfo(&params), "raw syevj info");
    int lwork = 0;
    check_solver(cusolverDnSsyevj_bufferSize(
                     cusolver_handle(), CUSOLVER_EIG_MODE_VECTOR,
                     CUBLAS_FILL_MODE_LOWER, side, d_a, side, d_w, &lwork,
                     params),
                 "raw syevj bufsize");
    checkCuda(cudaMalloc(&d_work, static_cast<std::size_t>(lwork) * sizeof(float)),
              "ext eigen work");
    check_solver(cusolverDnSsyevj(cusolver_handle(), CUSOLVER_EIG_MODE_VECTOR,
                                  CUBLAS_FILL_MODE_LOWER, side, d_a, side, d_w,
                                  d_work, lwork, d_info, params),
                 "raw syevj");
    cusolverDnDestroySyevjInfo(params);
    cudaFree(d_a);
    cudaFree(d_w);
    cudaFree(d_work);
    cudaFree(d_info);
}

void run_linalg_pairs(const Options& options, std::vector<Result>& out,
                      const std::vector<std::size_t>& sizes) {
    for (const std::size_t n : sizes) {
        if (n < 8) {
            continue;
        }
        if (!memory_available(16 * n * n * sizeof(float))) {
            if (!options.quiet) {
                std::cout << "  (skipping external-linalg " << n << "x" << n
                          << ": not enough free device memory)\n";
            }
            continue;
        }
        const std::string label = format_matrix_label(n, n);

        // --- solve: Matrix::solve vs raw getrf+getrs -------------------------
        {
            Matrix host_a = Matrix(n, n, host_spd_cm(n));
            Matrix host_b = Matrix(n, 1, std::vector<float>(n, 1.0f));
            const Stats mflash = sample_ms(
                [&] {
                    Matrix x = host_a.solve(host_b);
                    (void)x;
                },
                options);
            const double mflash_ops =
                mflash.median > 0.0 ? 1000.0 / mflash.median : 0.0;
            add_result(out, "external-linalg", label, n, mflash, mflash_ops,
                       "ops/s", "mflash solve (measured)");
            const Stats raw = sample_ms([n] { raw_solve_once(n); }, options);
            const double raw_ops = raw.median > 0.0 ? 1000.0 / raw.median : 0.0;
            add_result(out, "external-linalg", label, n, raw, raw_ops, "ops/s",
                       "raw cusolver getrf+getrs (measured)");
        }

        // --- qr: Matrix::qr vs raw geqrf --------------------------------------
        if (n <= 256) {
            Matrix host_m = Matrix(n, n, host_general_cm(n));
            const Stats mflash = sample_ms(
                [&] {
                    QRResult r = host_m.qr();
                    (void)r;
                },
                options);
            const double mflash_ops =
                mflash.median > 0.0 ? 1000.0 / mflash.median : 0.0;
            add_result(out, "external-linalg", label, n, mflash, mflash_ops,
                       "ops/s", "mflash qr (measured)");
            const Stats raw = sample_ms([n] { raw_qr_once(n); }, options);
            const double raw_ops = raw.median > 0.0 ? 1000.0 / raw.median : 0.0;
            add_result(out, "external-linalg", label, n, raw, raw_ops, "ops/s",
                       "raw cusolver geqrf (measured)");
        }

        // --- svd: Matrix::svd vs raw gesvdj (n <= 128: Jacobi is slow) --------
        if (n <= 128) {
            Matrix host_m = Matrix(n, n, host_general_cm(n));
            const Stats mflash = sample_ms(
                [&] {
                    SVDResult r = host_m.svd();
                    (void)r;
                },
                options);
            const double mflash_ops =
                mflash.median > 0.0 ? 1000.0 / mflash.median : 0.0;
            add_result(out, "external-linalg", label, n, mflash, mflash_ops,
                       "ops/s", "mflash svd (measured)");
            const Stats raw = sample_ms([n] { raw_svd_once(n); }, options);
            const double raw_ops = raw.median > 0.0 ? 1000.0 / raw.median : 0.0;
            add_result(out, "external-linalg", label, n, raw, raw_ops, "ops/s",
                       "raw cusolver gesvdj (measured)");
        }

        // --- eigen: Matrix::eigen vs raw syevj (n <= 128) ---------------------
        if (n <= 128) {
            Matrix host_m = Matrix(n, n, host_spd_cm(n));
            const Stats mflash = sample_ms(
                [&] {
                    EigenResult r = host_m.eigen();
                    (void)r;
                },
                options);
            const double mflash_ops =
                mflash.median > 0.0 ? 1000.0 / mflash.median : 0.0;
            add_result(out, "external-linalg", label, n, mflash, mflash_ops,
                       "ops/s", "mflash eigen (measured)");
            const Stats raw = sample_ms([n] { raw_eigen_once(n); }, options);
            const double raw_ops = raw.median > 0.0 ? 1000.0 / raw.median : 0.0;
            add_result(out, "external-linalg", label, n, raw, raw_ops, "ops/s",
                       "raw cusolver syevj (measured)");
        }
    }
}

void run_reduce_pairs(const Options& options, std::vector<Result>& out,
                      const std::vector<std::size_t>& elem_sizes) {
    constexpr std::size_t kCols = 1024;
    for (const std::size_t elems_in : elem_sizes) {
        const std::size_t elems = elems_in / kCols * kCols;
        if (elems == 0) {
            continue;
        }
        if (!memory_available(3 * elems * sizeof(float))) {
            continue;
        }
        Matrix input(elems / kCols, kCols,
                     std::vector<float>(elems, 0.5f));
        const std::string label = format_size(elems) + " elems";
        const double gbps = to_gbps(elems * sizeof(float), 1.0);

        // mflash sum vs thrust::reduce (when the header is visible).
        {
            const Stats ms = sample_ms([&] { (void)matrix_pro::sum(input); },
                                       options);
            add_result(out, "external-linalg", label, elems, ms,
                       gbps / ms.median, "GB/s", "mflash sum (measured)");
        }
#ifdef MATRIX_PRO_EXTERNAL_HAS_THRUST
        {
            thrust::device_ptr<float> dev(input.device_data());
            const Stats ms = sample_ms(
                [&] {
                    const float total = thrust::reduce(thrust::device, dev,
                                                       dev + elems, 0.0f,
                                                       thrust::plus<float>());
                    volatile float sink = total;
                    (void)sink;
                },
                options);
            add_result(out, "external-linalg", label, elems, ms,
                       gbps / ms.median, "GB/s", "thrust::reduce (measured)");
        }
#else
        if (!options.quiet) {
            std::cout << "  (skipping thrust: <thrust/reduce.h> not found)\n";
        }
#endif
#ifdef MATRIX_PRO_EXTERNAL_HAS_CUB
        {
            float* d_out = nullptr;
            void* d_temp = nullptr;
            std::size_t temp_bytes = 0;
            checkCuda(cudaMalloc(&d_out, sizeof(float)), "ext cub out");
            cub::DeviceReduce::Sum(d_temp, temp_bytes, input.device_data(),
                                   d_out, static_cast<int>(elems),
                                   compute_stream());
            checkCuda(cudaMalloc(&d_temp, temp_bytes), "ext cub temp");
            const Stats ms = sample_ms(
                [&] {
                    cub::DeviceReduce::Sum(d_temp, temp_bytes,
                                           input.device_data(), d_out,
                                           static_cast<int>(elems),
                                           compute_stream());
                },
                options);
            cudaFree(d_temp);
            cudaFree(d_out);
            add_result(out, "external-linalg", label, elems, ms,
                       gbps / ms.median, "GB/s", "cub::DeviceReduce (measured)");
        }
#else
        if (!options.quiet) {
            std::cout << "  (skipping cub: <cub/cub.cuh> not found)\n";
        }
#endif
    }
}

void run_external_linalg(const Options& options, std::vector<Result>& out) {
    const std::vector<std::size_t> sizes = options.sizes.empty()
                                               ? std::vector<std::size_t>{64, 128, 256}
                                               : options.sizes;
    run_linalg_pairs(options, out, sizes);
    const std::vector<std::size_t> elems = options.sizes.empty()
                                               ? std::vector<std::size_t>{1u << 20, 4u << 20}
                                               : options.sizes;
    run_reduce_pairs(options, out, elems);
}

} // namespace

const BenchmarkInfo& external_linalg_benchmark() {
    static const BenchmarkInfo info{
        "external_linalg",
        "raw cuSOLVER (solve/qr/svd/eigen) + thrust/cub reductions vs mflash",
        run_external_linalg};
    return info;
}

} // namespace bench
} // namespace matrix_pro




