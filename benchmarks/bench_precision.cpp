// Precision study: throughput *and* accuracy of the three GEMM paths.
//
//   fp32          plain cuBLAS SGEMM
//   fp16 inputs   tensor-core path (matmul_half)
//   fp64 accum.   numerically robust path (matmul_double_accumulate)
//
// The note column reports the relative error measured on two full rows,
// computed in double precision on the host, so the speed/accuracy trade-off is
// visible in a single table.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

#include "benchmark_support.hpp"
#include "matrix_pro/core/matrix.hpp"
#include "matrix_pro/ops/precision.hpp"
#include "matrix_pro/ops/product.hpp"

namespace matrix_pro {
namespace bench {
namespace {

std::vector<std::size_t> default_sizes() {
    return {512, 1024, 2048};
}

// Deterministic, well-scaled values in [1, 2) -- no cancellation, no overflow
// in half precision, so the error figure isolates rounding.
Matrix make_matrix(std::size_t rows, std::size_t cols) {
    std::vector<float> values(rows * cols);
    for (std::size_t i = 0; i < rows * cols; ++i) {
        values[i] = 1.0f + static_cast<float>((i * 2654435761u) % 1024u) / 1024.0f;
    }
    return Matrix(rows, cols, values);
}

// Relative error of `product` on two rows, against a double-precision host dot
// product. Rows are chosen so the cost stays O(n^2) instead of O(n^3).
double relative_error(const Matrix& product,
                      const std::vector<float>& left_row,
                      const std::vector<float>& right,
                      std::size_t n,
                      std::size_t product_row) {
    double worst = 0.0;
    double scale = 0.0;
    for (std::size_t col = 0; col < n; ++col) {
        double reference = 0.0;
        for (std::size_t k = 0; k < n; ++k) {
            reference += static_cast<double>(left_row[k]) *
                         static_cast<double>(right[k * n + col]);
        }
        const double actual = static_cast<double>(product.at(product_row, col));
        worst = std::max(worst, std::fabs(actual - reference));
        scale = std::max(scale, std::fabs(reference));
    }
    return scale > 0.0 ? worst / scale : 0.0;
}

std::string format_error(double value) {
    if (value <= 0.0) {
        return "exact";
    }
    if (value < 1e-3) {
        return "rel.err <1e-3";
    }
    return "rel.err " + std::to_string(static_cast<int>(value * 100.0 + 0.5)) + "%";
}

void run(const Options& options_orig, std::vector<Result>& out) {
    Options options = options_orig;
    if (options.sizes.empty()) {
        options.sizes = default_sizes();
    }

    for (const std::size_t n : options.sizes) {
        if (n < 16) {
            continue;
        }
        if (!memory_available(4 * n * n * sizeof(float))) {
            if (!options.quiet) {
                std::cout << "  (skipping " << format_matrix_label(n, n)
                          << ": not enough free device memory)\n";
            }
            continue;
        }

        const Matrix left = make_matrix(n, n);
        const Matrix right = make_matrix(n, n);
        const std::string label = format_matrix_label(n, n);
        const double flops = 2.0 * static_cast<double>(n) * n * n;

        // Accuracy check is O(n^2): enabled up to 1024 to keep the run fast.
        // Row 0 of the product is compared against a double-precision host dot
        // product of row 0 of A, so both sides describe the same entries.
        const bool check_accuracy = n <= 1024;
        const std::vector<float> left_row =
            check_accuracy ? std::vector<float>(left.data().begin(),
                                                left.data().begin() + static_cast<std::ptrdiff_t>(n))
                           : std::vector<float>();

        {
            const Stats ms = sample_ms([&] {
                Matrix product = matrix_pro::multiply(left, right);
                (void)product;
            }, options);
            std::string note = "cuBLAS fp32";
            if (check_accuracy) {
                Matrix product = matrix_pro::multiply(left, right);
                product.download();
                note += ", " + format_error(
                                    relative_error(product, left_row, right.data(), n, 0));
            }
            add_result(out, "precision", label, n, ms, to_gflops(flops, ms.median), "GFLOPS", note);
        }

        {
            const Stats ms = sample_ms([&] {
                Matrix product = matrix_pro::matmul_half(left, right);
                (void)product;
            }, options);
            std::string note = "fp16 inputs / fp32 out";
            if (check_accuracy) {
                Matrix product = matrix_pro::matmul_half(left, right);
                product.download();
                note += ", " + format_error(
                                    relative_error(product, left_row, right.data(), n, 0));
            }
            add_result(out, "precision", label, n, ms, to_gflops(flops, ms.median), "GFLOPS", note);
        }

        {
            const Stats ms = sample_ms([&] {
                Matrix product = matrix_pro::matmul_double_accumulate(left, right);
                (void)product;
            }, options);
            std::string note = "fp64 accumulation";
            if (check_accuracy) {
                Matrix product = matrix_pro::matmul_double_accumulate(left, right);
                product.download();
                note += ", " + format_error(
                                    relative_error(product, left_row, right.data(), n, 0));
            }
            add_result(out, "precision", label, n, ms, to_gflops(flops, ms.median), "GFLOPS", note);
        }
    }
}

} // namespace

const BenchmarkInfo& precision_benchmark() {
    static const BenchmarkInfo info{"precision",
                                    "fp32 / fp16 / fp64-accum GEMM: throughput and accuracy",
                                    run};
    return info;
}

} // namespace bench
} // namespace matrix_pro