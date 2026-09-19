#include <cmath>
#include <cstddef>
#include <functional>
#include <iostream>
#include <vector>

#include "benchmark_support.hpp"
#include "matrix_pro/core/matrix.hpp"
#include "matrix_pro/ops/math.hpp"

namespace matrix_pro {
namespace bench {
namespace {

constexpr std::size_t kColumns = 1024;

Matrix shaped(std::size_t elements, float value) {
    const std::size_t rows = elements / kColumns;
    return Matrix(rows, kColumns, std::vector<float>(rows * kColumns, value));
}

void case_unary(const char* name,
                const Options& options,
                std::vector<Result>& out,
                const std::function<Matrix(const Matrix&)>& op) {
    std::vector<std::size_t> sizes = options.sizes.empty() ?
        std::vector<std::size_t>{1u << 20, 4u << 20, 16u << 20} : options.sizes;

    for (const std::size_t elements_in : sizes) {
        const std::size_t elements = elements_in / kColumns * kColumns;
        if (elements == 0) continue;
        const std::size_t bytes = 2 * elements * sizeof(float); // 1 read + 1 write
        if (!memory_available(bytes)) continue;

        Matrix operand = shaped(elements, 1.25f);
        const Stats ms = sample_ms([&] { op(operand); }, options);
        add_result(out, "math", format_size(elements) + " elems", elements, ms,
                   to_gbps(bytes, ms.median), "GB/s", name);
    }
}

void run(const Options& options, std::vector<Result>& out) {
    case_unary("sin", options, out, [](const Matrix& a) { return matrix_pro::sin(a); });
    case_unary("cos", options, out, [](const Matrix& a) { return matrix_pro::cos(a); });
    case_unary("erf", options, out, [](const Matrix& a) { return matrix_pro::erf(a); });
    case_unary("rsqrt", options, out, [](const Matrix& a) { return matrix_pro::rsqrt(a); });
    case_unary("reciprocal", options, out, [](const Matrix& a) { return matrix_pro::reciprocal(a); });
}

} // namespace

const BenchmarkInfo& math_benchmark() {
    static const BenchmarkInfo info{"math",
                                    "extended math ops: effective GB/s throughput",
                                    run};
    return info;
}

} // namespace bench
} // namespace matrix_pro

