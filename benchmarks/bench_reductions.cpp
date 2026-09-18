// Reduction throughput: scalar reductions and axis reductions.
//
// These kernels are pure global-memory passes with a tiny output, so the
// effective GB/s figure is the meaningful number: it should approach the
// device's peak bandwidth for the large cases.

#include <cstddef>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "benchmark_support.hpp"
#include "matrix_pro/core/matrix.hpp"
#include "matrix_pro/ops/reductions.hpp"

namespace matrix_pro {
namespace bench {
namespace {

constexpr std::size_t kColumns = 1024;

std::vector<std::size_t> default_sizes() {
    return {1u << 20, 4u << 20, 16u << 20};
}

Matrix shaped(std::size_t elements) {
    const std::size_t rows = elements / kColumns;
    std::vector<float> values(rows * kColumns);
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>((i % 251) + 1) * 0.25f;   // 0.25 .. 63.0, non-zero
    }
    return Matrix(rows, kColumns, values);
}

void run(const Options& options_orig, std::vector<Result>& out) {
    Options options = options_orig;
    if (options.sizes.empty()) {
        options.sizes = default_sizes();
    }

    for (const std::size_t elements_in : options.sizes) {
        const std::size_t elements = elements_in / kColumns * kColumns;
        if (elements == 0) {
            continue;
        }
        const std::size_t bytes = elements * sizeof(float);
        if (!memory_available(bytes + elements * sizeof(float))) {
            if (!options.quiet) {
                std::cout << "  (skipping " << format_size(elements_in)
                          << ": needs " << format_size(bytes) << "B)\n";
            }
            continue;
        }

        Matrix input = shaped(elements);
        const std::string label = format_size(elements) + " elems";
        const double gbps = to_gbps(bytes, 1.0);

        struct Case {
            const char* name;
            std::function<void()> call;
        };

        std::vector<Case> cases;
        cases.push_back({"sum", [&input] { (void)matrix_pro::sum(input); }});
        cases.push_back({"mean", [&input] { (void)matrix_pro::mean(input); }});
        cases.push_back({"min", [&input] { (void)matrix_pro::min(input); }});
        cases.push_back({"max", [&input] { (void)matrix_pro::max(input); }});
        cases.push_back({"l1_norm", [&input] { (void)matrix_pro::l1_norm(input); }});
        cases.push_back({"l2_norm", [&input] { (void)matrix_pro::l2_norm(input); }});
        cases.push_back({"abs_max", [&input] { (void)matrix_pro::abs_max(input); }});
        cases.push_back({"argmax", [&input] { (void)matrix_pro::argmax(input); }});
        cases.push_back({"stddev", [&input] { (void)matrix_pro::stddev(input); }});
        cases.push_back({"row_sum", [&input] { (void)matrix_pro::row_sum(input); }});
        cases.push_back({"col_sum", [&input] { (void)matrix_pro::col_sum(input); }});

        for (Case& item : cases) {
            const Stats ms = sample_ms(item.call, options);
            add_result(out, "reductions", label, elements, ms,
                       gbps / ms.median, "GB/s", item.name);
        }
    }
}

} // namespace

const BenchmarkInfo& reductions_benchmark() {
    static const BenchmarkInfo info{"reductions",
                                    "scalar / axis reductions: effective GB/s per kernel",
                                    run};
    return info;
}

} // namespace bench
} // namespace matrix_pro
