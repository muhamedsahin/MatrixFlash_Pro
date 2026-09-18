// Elementwise throughput: the memory-bound half of the library.
//
// Every case reports effective GB/s, derived from the actual traffic of the
// operation (read operands + write result) so the numbers can be compared
// directly against the theoretical bandwidth of the GPU.

#include <cmath>
#include <cstddef>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "benchmark_support.hpp"
#include "matrix_pro/core/matrix.hpp"
#include "matrix_pro/nn/inplace.hpp"
#include "matrix_pro/ops/elementwise.hpp"

namespace matrix_pro {
namespace bench {
namespace {

constexpr std::size_t kColumns = 1024;

std::vector<std::size_t> default_sizes() {
    return {1u << 20, 4u << 20, 16u << 20};   // 1M, 4M, 16M elements
}

// Shapes an element count as (rows x 1024) so the memory access pattern stays
// realistic (multiple rows, contiguous within a row).
Matrix shaped(std::size_t elements, float value) {
    const std::size_t rows = elements / kColumns;
    return Matrix(rows, kColumns, std::vector<float>(rows * kColumns, value));
}

void case_binary(const char* name,
                 const Options& options,
                 std::vector<Result>& out,
                 const std::function<Matrix(const Matrix&, const Matrix&)>& op) {
    for (const std::size_t elements_in : options.sizes) {
        const std::size_t elements = elements_in / kColumns * kColumns;
        if (elements == 0) {
            continue;
        }
        const std::size_t bytes = 3 * elements * sizeof(float);
        if (!memory_available(bytes)) {
            if (!options.quiet) {
                std::cout << "  (skipping " << name << " " << format_size(elements_in)
                          << ": needs " << format_size(bytes) << "B)\n";
            }
            continue;
        }

        Matrix left = shaped(elements, 1.25f);
        Matrix right = shaped(elements, 0.75f);

        Matrix reference = op(left, right);
        reference.download();
        if (std::fabs(reference.at(0, 0)) < 1e-6f) {
            throw std::runtime_error(std::string("elementwise validation failed: ") + name);
        }
        reference = Matrix();

        const Stats ms = sample_ms([&] {
            Matrix result = op(left, right);
            (void)result;
        }, options);

        add_result(out, "elementwise", format_size(elements) + " elems", elements, ms,
                   to_gbps(bytes, ms.median), "GB/s", name);
    }
}

void case_unary(const char* name,
                const Options& options,
                std::vector<Result>& out,
                const std::function<Matrix(const Matrix&)>& op) {
    for (const std::size_t elements_in : options.sizes) {
        const std::size_t elements = elements_in / kColumns * kColumns;
        if (elements == 0) {
            continue;
        }
        const std::size_t bytes = 2 * elements * sizeof(float);
        if (!memory_available(bytes)) {
            continue;
        }

        Matrix input = shaped(elements, 0.5f);

        Matrix reference = op(input);
        reference.download();
        if (std::fabs(reference.at(0, 0)) < 1e-9f) {
            throw std::runtime_error(std::string("unary validation failed: ") + name);
        }
        reference = Matrix();

        const Stats ms = sample_ms([&] {
            Matrix result = op(input);
            (void)result;
        }, options);

        add_result(out, "elementwise", format_size(elements) + " elems", elements, ms,
                   to_gbps(bytes, ms.median), "GB/s", name);
    }
}

void run(const Options& options_orig, std::vector<Result>& out) {
    Options options = options_orig;
    if (options.sizes.empty()) {
        options.sizes = default_sizes();
    }

    // --- binary ops ----------------------------------------------------------
    case_binary("add", options, out, [](const Matrix& a, const Matrix& b) {
        return matrix_pro::add(a, b);
    });
    case_binary("hadamard", options, out, [](const Matrix& a, const Matrix& b) {
        return matrix_pro::elementwise_multiply(a, b);
    });
    case_binary("divide", options, out, [](const Matrix& a, const Matrix& b) {
        return matrix_pro::divide(a, b);
    });

    // --- unary ops (read + write) -------------------------------------------
    case_unary("scale(*1.5)", options, out, [](const Matrix& a) {
        return matrix_pro::multiply(a, 1.5f);
    });
    case_unary("add_scalar", options, out, [](const Matrix& a) {
        return matrix_pro::add_scalar(a, 2.0f);
    });
    case_unary("exp", options, out, [](const Matrix& a) {
        return matrix_pro::exp(a);
    });

    // --- in-place: same traffic, no allocation, no extra memory cost --------
    for (const std::size_t elements_in : options.sizes) {
        const std::size_t elements = elements_in / kColumns * kColumns;
        if (elements == 0) {
            continue;
        }
        if (!memory_available(2 * elements * sizeof(float))) {
            continue;
        }
        Matrix accumulator = shaped(elements, 1.0f);
        Matrix operand = shaped(elements, 0.5f);
        const std::size_t bytes = 3 * elements * sizeof(float);

        const Stats ms = sample_ms([&] { matrix_pro::add_(accumulator, operand); }, options);
        add_result(out, "elementwise", format_size(elements) + " elems", elements, ms,
                   to_gbps(bytes, ms.median), "GB/s", "add_ (in-place)");
    }
}

} // namespace

const BenchmarkInfo& elementwise_benchmark() {
    static const BenchmarkInfo info{"elementwise",
                                    "elementwise arithmetic: effective GB/s vs GPU bandwidth",
                                    run};
    return info;
}

} // namespace bench
} // namespace matrix_pro
