// MatrixFlash-Pro vs same-machine baselines (GEMM + MLP tape cost).
// Full header comment: see git history / docs page for methodology.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "benchmark_support.hpp"
#include "matrix_pro/autograd/autograd.hpp"
#include "matrix_pro/core/cuda_utils.hpp"
#include "matrix_pro/core/matrix.hpp"
#include "matrix_pro/ops/product.hpp"
#include "matrix_pro/ops/shape.hpp"

#if defined(__has_include)
#if __has_include(<Eigen/Dense>)
#include <Eigen/Dense>
#define MATRIX_PRO_COMPARISON_HAS_EIGEN 1
#endif
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace matrix_pro {
namespace bench {
namespace {

// Host wall-clock sampler for CPU baselines (CUDA events measure device
// time only and would report ~0 for host work).
Stats cpu_sample_ms(const std::function<void()>& fn, const Options& options) {
    const int warmup = std::max(0, options.warmup);
    const int repeats = std::max(1, options.repeats);
    for (int i = 0; i < warmup; ++i) {
        fn();
    }
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeats));
    for (int i = 0; i < repeats; ++i) {
        const auto start = std::chrono::high_resolution_clock::now();
        fn();
        const auto stop = std::chrono::high_resolution_clock::now();
        samples.push_back(
            std::chrono::duration<double, std::milli>(stop - start).count());
    }
    std::sort(samples.begin(), samples.end());
    Stats stats;
    stats.min = samples.front();
    stats.max = samples.back();
    const std::size_t mid = samples.size() / 2;
    stats.median = (samples.size() % 2 == 0)
                       ? 0.5 * (samples[mid - 1] + samples[mid])
                       : samples[mid];
    const double total =
        std::accumulate(samples.begin(), samples.end(), 0.0);
    stats.mean = total / static_cast<double>(samples.size());
    double variance = 0.0;
    for (double value : samples) {
        variance += (value - stats.mean) * (value - stats.mean);
    }
    stats.stddev = std::sqrt(variance / static_cast<double>(samples.size()));
    return stats;
}

// Naive global-memory GEMM kernel (row-major, no tiling/shared memory).
__global__ void naive_gemm_kernel(const float* left, const float* right,
                                  float* output, int n) {
    const int col = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int row = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (row >= n || col >= n) {
        return;
    }
    float acc = 0.0f;
    for (int k = 0; k < n; ++k) {
        acc += left[row * n + k] * right[k * n + col];
    }
    output[row * n + col] = acc;
}

std::vector<std::size_t> default_comparison_sizes() {
    return {256, 512, 1024, 2048};
}

void run_gemm_cases(const Options& options, std::vector<Result>& out,
                    const std::vector<std::size_t>& sizes) {
    for (const std::size_t n : sizes) {
        if (n < 32) {
            continue;
        }
        const std::size_t bytes = 3 * n * n * sizeof(float);
        if (!memory_available(bytes * 2)) {
            if (!options.quiet) {
                std::cout << "  (skipping comparison " << n << "x" << n
                          << ": not enough free device memory)\n";
            }
            continue;
        }
        const std::string label = format_matrix_label(n, n);
        const double flops = 2.0 * static_cast<double>(n) * n * n;

        // (a) MatrixFlash-Pro operator* (alloc included: what apps pay).
        {
            Matrix left = Matrix::ones(n, n);
            Matrix right = Matrix::ones(n, n);
            Matrix reference = left * right;
            reference.download();
            const float expected = static_cast<float>(n);
            if (std::fabs(reference.at(0, 0) - expected) > 1.0f) {
                throw std::runtime_error("comparison validation (mflash)");
            }
            reference = Matrix();
            const Stats ms = sample_ms(
                [&left, &right] {
                    Matrix product = left * right;
                    (void)product;
                },
                options);
            add_result(out, "comparison", label, n, ms,
                       to_gflops(flops, ms.median), "GFLOPS",
                       "mflash operator* (measured)");
        }

        // (b) Raw cuBLAS on pre-allocated buffers (no alloc in timing).
        {
            float* raw_left = nullptr;
            float* raw_right = nullptr;
            float* raw_out = nullptr;
            checkCuda(cudaMalloc(&raw_left, n * n * sizeof(float)),
                      "comparison cudaMalloc");
            checkCuda(cudaMalloc(&raw_right, n * n * sizeof(float)),
                      "comparison cudaMalloc");
            checkCuda(cudaMalloc(&raw_out, n * n * sizeof(float)),
                      "comparison cudaMalloc");
            checkCuda(cudaMemset(raw_left, 0, n * n * sizeof(float)),
                      "comparison cudaMemset");
            checkCuda(cudaMemset(raw_right, 0, n * n * sizeof(float)),
                      "comparison cudaMemset");
            const float alpha = 1.0f;
            const float beta = 0.0f;
            cublasHandle_t handle = cublas_handle();
            cublasSetStream(handle, compute_stream());
            const int side = static_cast<int>(n);
            const Stats ms = sample_ms(
                [&] {
                    const cublasStatus_t status = cublasGemmEx(
                        handle, CUBLAS_OP_N, CUBLAS_OP_N, side, side, side,
                        &alpha, raw_right, CUDA_R_32F, side, raw_left,
                        CUDA_R_32F, side, &beta, raw_out, CUDA_R_32F, side,
                        CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
                    if (status != CUBLAS_STATUS_SUCCESS) {
                        throw std::runtime_error("raw cuBLAS gemm failed");
                    }
                },
                options);
            cudaFree(raw_left);
            cudaFree(raw_right);
            cudaFree(raw_out);
            add_result(out, "comparison", label, n, ms,
                       to_gflops(flops, ms.median), "GFLOPS",
                       "raw cublasGemmEx, no alloc (measured)");
        }

        // (c) Naive global-memory kernel (n <= 1024 only).
        if (n <= 1024) {
            Matrix left = Matrix::ones(n, n);
            Matrix right = Matrix::ones(n, n);
            Matrix output(n, n);
            dim3 block(16, 16);
            dim3 grid((static_cast<unsigned int>(n) + 15) / 16,
                      (static_cast<unsigned int>(n) + 15) / 16);
            const int side = static_cast<int>(n);
            const Stats ms = sample_ms(
                [&] {
                    naive_gemm_kernel<<<grid, block, 0, compute_stream()>>>(
                        left.device_data(), right.device_data(),
                        output.device_data(), side);
                },
                options);
            checkCuda(cudaGetLastError(), "naive gemm kernel launch");
            add_result(out, "comparison", label, n, ms,
                       to_gflops(flops, ms.median), "GFLOPS",
                       "naive global-mem kernel (measured)");
        } else if (!options.quiet) {
            std::cout << "  (skipping naive kernel at " << n << "x" << n
                      << ": O(n^3) thread work)\n";
        }
        // (d) CPU single-thread naive (host wall-clock, n <= 512 only).
        if (n <= 512) {
            std::vector<float> host_left(n * n, 1.0f);
            std::vector<float> host_right(n * n, 1.0f);
            std::vector<float> host_out(n * n, 0.0f);
            const Stats ms = cpu_sample_ms(
                [&] {
                    for (std::size_t i = 0; i < n; ++i) {
                        for (std::size_t k = 0; k < n; ++k) {
                            const float aik = host_left[i * n + k];
                            for (std::size_t j = 0; j < n; ++j) {
                                host_out[i * n + j] +=
                                    aik * host_right[k * n + j];
                            }
                        }
                    }
                    volatile float sink = host_out[0];
                    (void)sink;
                    std::fill(host_out.begin(), host_out.end(), 0.0f);
                },
                options);
            if (std::fabs(host_out[0]) > 0.5f) {
                throw std::runtime_error("cpu baseline validation failed");
            }
            add_result(out, "comparison", label, n, ms,
                       to_gflops(flops, ms.median), "GFLOPS",
                       "cpu single-thread naive, wall-clock (measured)");
        } else if (!options.quiet) {
            std::cout << "  (skipping cpu naive at " << n << "x" << n
                      << ": O(n^3) host loop)\n";
        }

        // (e) CPU OpenMP blocked (host wall-clock, n <= 512 only).
#ifdef _OPENMP
        if (n <= 512) {
            std::vector<float> host_left(n * n, 1.0f);
            std::vector<float> host_right(n * n, 1.0f);
            std::vector<float> host_out(n * n, 0.0f);
            const Stats ms = cpu_sample_ms(
                [&] {
#pragma omp parallel for schedule(static)
                    for (long long ii = 0; ii < static_cast<long long>(n); ++ii) {
                        const std::size_t i = static_cast<std::size_t>(ii);
                        for (std::size_t k = 0; k < n; ++k) {
                            const float aik = host_left[i * n + k];
                            for (std::size_t j = 0; j < n; ++j) {
                                host_out[i * n + j] +=
                                    aik * host_right[k * n + j];
                            }
                        }
                    }
                    volatile float sink = host_out[0];
                    (void)sink;
                    std::fill(host_out.begin(), host_out.end(), 0.0f);
                },
                options);
            add_result(out, "comparison", label, n, ms,
                       to_gflops(flops, ms.median), "GFLOPS",
                       "cpu OpenMP blocked, wall-clock (measured)");
        }
#else
        if (!options.quiet) {
            std::cout << "  (skipping cpu-omp: built without OpenMP)\n";
        }
#endif

        // (f) Eigen, only when the header was visible at build time.
#ifdef MATRIX_PRO_COMPARISON_HAS_EIGEN
        if (n <= 512) {
            Eigen::MatrixXf eigen_left = Eigen::MatrixXf::Ones(
                static_cast<int>(n), static_cast<int>(n));
            Eigen::MatrixXf eigen_right = Eigen::MatrixXf::Ones(
                static_cast<int>(n), static_cast<int>(n));
            const Stats ms = cpu_sample_ms(
                [&] {
                    Eigen::MatrixXf product = eigen_left * eigen_right;
                    volatile float sink = product(0, 0);
                    (void)sink;
                },
                options);
            add_result(out, "comparison", label, n, ms,
                       to_gflops(flops, ms.median), "GFLOPS",
                       "eigen sgemm, wall-clock (measured)");
        }
#else
        if (!options.quiet) {
            std::cout << "  (skipping eigen: Eigen/Dense not at build)\n";
        }
#endif

    }
}

constexpr std::size_t kMlpBatch = 64;
constexpr std::size_t kMlpInputs = 256;
constexpr std::size_t kMlpClasses = 10;

void run_mlp_cases(const Options& options, std::vector<Result>& out,
                   const std::vector<std::size_t>& sizes) {
    std::vector<float> labels(kMlpBatch);
    for (std::size_t i = 0; i < kMlpBatch; ++i) {
        labels[i] = static_cast<float>(i % kMlpClasses);
    }
    const Matrix targets =
        matrix_pro::one_hot(Matrix(kMlpBatch, 1, labels), kMlpClasses);
    for (const std::size_t hidden : sizes) {
        if (hidden < 4 || hidden > 2048) {
            continue;
        }
        if (!memory_available(
                8 * (hidden * kMlpInputs + kMlpBatch * hidden) * sizeof(float))) {
            continue;
        }
        Variable input(Matrix::uniform(kMlpBatch, kMlpInputs, -0.5f, 0.5f), false);
        Variable target(targets, false);
        Variable w1(Matrix::glorot(kMlpInputs, hidden));
        Variable b1(Matrix::zeros(1, hidden));
        Variable w2(Matrix::glorot(hidden, kMlpClasses));
        Variable b2(Matrix::zeros(1, kMlpClasses));
        const std::function<void()> full_step = [&] {
            w1.zero_grad();
            b1.zero_grad();
            w2.zero_grad();
            b2.zero_grad();
            Variable h = input.matmul(w1).broadcast_add(b1).relu();
            Variable logits = h.matmul(w2).broadcast_add(b2);
            Variable loss = softmax_cross_entropy_loss(logits, target);
            loss.backward();
        };
        full_step();
        const Stats full_ms = sample_ms(full_step, options);
        const double full_sps = full_ms.median > 0.0
                                    ? static_cast<double>(kMlpBatch) * 1000.0 / full_ms.median
                                    : 0.0;
        add_result(out, "comparison", "mlp-full h=" + std::to_string(hidden), hidden,
                   full_ms, full_sps, "samples/s",
                   "autograd fwd+bwd, batch=64 (measured)");
        const std::function<void()> fwd_only = [&] {
            Variable h = input.matmul(w1).broadcast_add(b1).relu();
            Variable logits = h.matmul(w2).broadcast_add(b2);
            Variable loss = softmax_cross_entropy_loss(logits, target);
            (void)loss;
        };
        const Stats fwd_ms = sample_ms(fwd_only, options);
        const double fwd_sps = fwd_ms.median > 0.0
                                   ? static_cast<double>(kMlpBatch) * 1000.0 / fwd_ms.median
                                   : 0.0;
        const double tape_cost = fwd_ms.median > 0.0 ? full_ms.median / fwd_ms.median : 0.0;
        const long long tenths = static_cast<long long>(tape_cost * 10.0 + 0.5);
        add_result(out, "comparison", "mlp-fwd h=" + std::to_string(hidden), hidden,
                   fwd_ms, fwd_sps, "samples/s",
                   "forward only, tape cost " + std::to_string(tenths / 10) + "." +
                       std::to_string(tenths % 10) + "x (measured)");
    }
}

void run_gemm_and_mlp(const Options& options, std::vector<Result>& out) {
    const std::vector<std::size_t> sizes =
        options.sizes.empty() ? default_comparison_sizes() : options.sizes;
    run_gemm_cases(options, out, sizes);
    const std::vector<std::size_t> mlp_sizes =
        options.sizes.empty() ? std::vector<std::size_t>{128, 256, 512} : options.sizes;
    run_mlp_cases(options, out, mlp_sizes);
}

} // namespace

const BenchmarkInfo& comparison_benchmark() {
    static const BenchmarkInfo info{
        "comparison",
        "same-machine baselines: mflash vs raw cuBLAS vs naive GPU/CPU vs Eigen(opt) + MLP tape cost",
        run_gemm_and_mlp};
    return info;
}

} // namespace bench
} // namespace matrix_pro

