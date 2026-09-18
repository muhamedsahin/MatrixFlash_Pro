// Activations: the standard elementwise activations, their in-place variants
// and the fused chains, so the cost of an extra global-memory round trip is
// directly visible in the table.
//
//   "relu"            -> out = relu(x)                 (read + write)
//   "relu_ (inplace)" -> x = relu(x)                   (read + write, no alloc)
//   "fused s(x)*y"    -> fused_sigmoid_mul(x, y)       (3 passes of traffic)
//   "chain s(x)*y"    -> sigmoid(x); multiply(t, y)    (5 passes of traffic)

#include <cmath>
#include <cstddef>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "benchmark_support.hpp"
#include "matrix_pro/core/matrix.hpp"
#include "matrix_pro/nn/fused.hpp"
#include "matrix_pro/nn/inplace.hpp"
#include "matrix_pro/ops/elementwise.hpp"
#include "matrix_pro/ops/transforms.hpp"

namespace matrix_pro {
namespace bench {
namespace {

constexpr std::size_t kColumns = 1024;

std::vector<std::size_t> default_sizes() {
    return {1u << 20, 4u << 20, 16u << 20};
}

Matrix shaped(std::size_t elements, float value) {
    const std::size_t rows = elements / kColumns;
    return Matrix(rows, kColumns, std::vector<float>(rows * kColumns, value));
}

void unary_case(const char* name,
                const std::size_t elements,
                const Options& options,
                std::vector<Result>& out,
                const std::function<Matrix()>& op) {
    const std::size_t bytes = 2 * elements * sizeof(float);
    const Stats ms = sample_ms(op, options);
    add_result(out, "activations", format_size(elements) + " elems", elements, ms,
               to_gbps(bytes, ms.median), "GB/s", name);
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
        if (!memory_available(3 * elements * sizeof(float))) {
            if (!options.quiet) {
                std::cout << "  (skipping " << format_size(elements_in)
                          << ": needs " << format_size(3 * elements * sizeof(float)) << "B)\n";
            }
            continue;
        }

        // Small magnitudes keep every activation in its linear region: no
        // saturating / NaN corner cases while measuring throughput.
        Matrix input = shaped(elements, 0.35f);

        unary_case("relu", elements, options, out, [&input] { return matrix_pro::relu(input); });
        unary_case("sigmoid", elements, options, out, [&input] { return matrix_pro::sigmoid(input); });
        unary_case("tanh", elements, options, out, [&input] { return matrix_pro::tanh(input); });
        unary_case("gelu", elements, options, out, [&input] { return matrix_pro::gelu(input); });
        unary_case("swish", elements, options, out, [&input] { return matrix_pro::swish(input); });
        unary_case("elu", elements, options, out, [&input] { return matrix_pro::elu(input); });
        unary_case("softmax", elements, options, out, [&input] { return matrix_pro::softmax(input); });

        // --- in-place vs out-of-place (same traffic, one less allocation) -----
        {
            Matrix buffer = input;
            const Stats ms = sample_ms([&buffer] { matrix_pro::relu_(buffer); }, options);
            add_result(out, "activations", format_size(elements) + " elems", elements, ms,
                       to_gbps(2 * elements * sizeof(float), ms.median), "GB/s",
                       "relu_ (in-place)");
        }

        // --- fused chain vs unfused chain ------------------------------------
        {
            Matrix other = shaped(elements, 0.75f);

            const Stats fused = sample_ms([&input, &other] {
                Matrix value = matrix_pro::fused_sigmoid_mul(input, other);
                (void)value;
            }, options);

            const Stats chained = sample_ms([&input, &other] {
                Matrix activated = matrix_pro::sigmoid(input);
                Matrix value = matrix_pro::elementwise_multiply(activated, other);
                (void)value;
            }, options);

            const double speedup = fused.median > 0.0 ? chained.median / fused.median : 0.0;
            add_result(out, "activations", format_size(elements) + " elems", elements, fused,
                       to_gbps(3 * elements * sizeof(float), fused.median), "GB/s",
                       "fused sigmoid(x)*y");
            add_result(out, "activations", format_size(elements) + " elems", elements, chained,
                       to_gbps(5 * elements * sizeof(float), chained.median), "GB/s",
                       "chain sigmoid + mul");
            add_result(out, "activations", format_size(elements) + " elems", elements, fused,
                       to_gbps(5 * elements * sizeof(float), fused.median), "GB/s",
                       "fused speedup " +
                           std::to_string(static_cast<long long>(speedup * 100.0 + 0.5)) + "%");
        }
    }
}

} // namespace

const BenchmarkInfo& activations_benchmark() {
    static const BenchmarkInfo info{"activations",
                                    "activation kernels: out-of-place vs in-place vs fused chains",
                                    run};
    return info;
}

} // namespace bench
} // namespace matrix_pro
