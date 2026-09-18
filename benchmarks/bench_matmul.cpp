// Dense GEMM throughput (cuBLAS) across square and rectangular shapes.
//
// Case labels
//   "N x N"       -> (N x N) * (N x N)
//   "M x K x N"   -> (M x K) * (K x N), with K = N and M = N/2
//
// Metrics: device milliseconds (CUDA events) and GFLOPS = 2*M*K*N / time.
// The allocation of the output matrix is part of the measurement, because that
// is what an application actually pays per call.

#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "benchmark_support.hpp"
#include "matrix_pro/core/matrix.hpp"
#include "matrix_pro/ops/product.hpp"

namespace matrix_pro {
namespace bench {
namespace {

constexpr std::size_t kMinimumSize = 32;

std::vector<std::size_t> default_sizes() {
    return {256, 512, 1024, 2048};
}

std::string rectangular_label(std::size_t m, std::size_t k, std::size_t n) {
    return format_matrix_label(m, k) + "x" + std::to_string(n);
}

void run(const Options& options, std::vector<Result>& out) {
    const std::vector<std::size_t> sizes = options.sizes.empty() ? default_sizes() : options.sizes;

    for (const std::size_t n : sizes) {
        if (n < kMinimumSize) {
            continue;
        }

        // --- square GEMM -----------------------------------------------------
        const std::size_t square_bytes = 3 * n * n * sizeof(float);
        if (!memory_available(square_bytes)) {
            if (!options.quiet) {
                std::cout << "  (skipping " << format_matrix_label(n, n)
                          << ": needs " << format_size(square_bytes) << "B)\n";
            }
        } else {
            Matrix left = Matrix::ones(n, n);
            Matrix right = Matrix::ones(n, n);

            // Correctness gate: ones * ones = n everywhere.
            Matrix reference = left * right;
            reference.download();
            const float expected = static_cast<float>(n);
            if (std::fabs(reference.at(0, 0) - expected) > 1.0f ||
                std::fabs(reference.at(n - 1, n - 1) - expected) > 1.0f) {
                throw std::runtime_error("matmul validation failed for " +
                                         format_matrix_label(n, n));
            }
            reference = Matrix();

            const double flops = 2.0 * static_cast<double>(n) * n * n;
            const Stats ms = sample_ms([&left, &right] {
                Matrix product = left * right;
                (void)product;
            }, options);

            add_result(out, "matmul", format_matrix_label(n, n), n, ms,
                       to_gflops(flops, ms.median), "GFLOPS",
                       "square, cuBLAS fp32");
        }

        // --- rectangular GEMM -------------------------------------------------
        const std::size_t m = n / 2;
        if (m < kMinimumSize) {
            continue;
        }
        const std::size_t rect_bytes = (m * n + n * n + m * n) * sizeof(float);
        if (!memory_available(rect_bytes)) {
            continue;
        }

        Matrix tall = Matrix::ones(m, n);
        Matrix wide = Matrix::ones(n, n);
        Matrix reference = tall * wide;
        reference.download();
        if (std::fabs(reference.at(0, 0) - static_cast<float>(n)) > 1.0f) {
            throw std::runtime_error("rectangular matmul validation failed");
        }
        reference = Matrix();

        const double rect_flops = 2.0 * static_cast<double>(m) * n * n;
        const Stats rect_ms = sample_ms([&tall, &wide] {
            Matrix product = tall * wide;
            (void)product;
        }, options);

        add_result(out, "matmul", rectangular_label(m, n, n), n, rect_ms,
                   to_gflops(rect_flops, rect_ms.median), "GFLOPS",
                   "rectangular M x K x N");
    }
}

} // namespace

const BenchmarkInfo& matmul_benchmark() {
    static const BenchmarkInfo info{"matmul",
                                    "dense GEMM (cuBLAS): GFLOPS across square / rectangular shapes",
                                    run};
    return info;
}

} // namespace bench
} // namespace matrix_pro
