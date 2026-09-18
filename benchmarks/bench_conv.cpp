// Convolution / pooling throughput (NCHW), forward and backward.
//
// The library uses cuDNN when it is available and otherwise falls back to its
// own CUDA kernels; the table records which backend was used in the note
// column, so results from different machines stay comparable.
//
// The `size` CLI value is interpreted as the square spatial edge H = W of the
// input image.

#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "benchmark_support.hpp"
#include "matrix_pro/core/tensor.hpp"
#include "matrix_pro/nn/conv.hpp"

namespace matrix_pro {
namespace bench {
namespace {

std::vector<std::size_t> default_sizes() {
    return {64, 128, 224};
}

Tensor filled(const std::vector<std::size_t>& shape, float value) {
    Tensor tensor(shape);
    tensor.fill(value);
    return tensor;
}

std::string backend_note() {
#if defined(MATRIX_PRO_HAS_CUDNN)
    return "cuDNN";
#else
    return "native CUDA fallback";
#endif
}

// Element count of an NCHW tensor description.
std::size_t elements(std::size_t n, std::size_t c, std::size_t h, std::size_t w) {
    return n * c * h * w;
}

void run(const Options& options_orig, std::vector<Result>& out) {
    Options options = options_orig;
    if (options.sizes.empty()) {
        options.sizes = default_sizes();
    }

    const std::string backend = backend_note();

    for (const std::size_t edge : options.sizes) {
        if (edge < 16) {
            continue;
        }
        const std::size_t half = edge / 2;

        // --- conv2d 3 -> 16, 3x3, stride 1, no padding -----------------------
        {
            const std::size_t bytes = (elements(1, 3, edge, edge) +
                                       elements(16, 3, 3, 3) + 16) * sizeof(float);
            if (!memory_available(bytes * 4)) {
                if (!options.quiet) {
                    std::cout << "  (skipping conv 3->16 at " << edge
                              << ": not enough free device memory)\n";
                }
            } else {
                Tensor input = filled({1, 3, edge, edge}, 0.1f);
                Tensor weights = filled({16, 3, 3, 3}, 0.05f);
                Tensor bias = filled({16}, 0.0f);

                Tensor reference = matrix_pro::conv2d(input, weights, bias, 1, 0);
                reference.download();
                for (const float value : reference.data()) {
                    if (!std::isfinite(value)) {
                        throw std::runtime_error("conv2d produced a non-finite value");
                    }
                }

                const std::size_t out_edge = edge - 2;   // 3x3 kernel, no padding
                const double flops =
                    2.0 * 16 * 3 * 9 * static_cast<double>(out_edge) * out_edge;
                const Stats ms = sample_ms([&] {
                    Tensor result = matrix_pro::conv2d(input, weights, bias, 1, 0);
                    (void)result;
                }, options);
                add_result(out, "conv", "conv2d 3->16 " + std::to_string(edge), edge, ms,
                           to_gflops(flops, ms.median), "GFLOPS", backend);

                // Backward passes only for the small configurations: the
                // fallback kernels are compute-heavy.
                if (edge <= 128) {
                    const Stats input_grad = sample_ms([&] {
                        Tensor grad = matrix_pro::conv2d_input_backward(reference, input, weights, 1, 0);
                        (void)grad;
                    }, options);
                    add_result(out, "conv", "conv2d dInput " + std::to_string(edge), edge,
                               input_grad, to_gflops(flops, input_grad.median), "GFLOPS", backend);

                    const Stats weight_grad = sample_ms([&] {
                        Tensor grad = matrix_pro::conv2d_weight_backward(reference, input, weights, 1, 0);
                        (void)grad;
                    }, options);
                    add_result(out, "conv", "conv2d dWeight " + std::to_string(edge), edge,
                               weight_grad, to_gflops(flops, weight_grad.median), "GFLOPS", backend);
                }
            }
        }

        // --- conv2d 32 -> 64, 3x3, stride 1, padding 1 (typical CNN block) ----
        {
            const std::size_t bytes = (elements(1, 32, half, half) +
                                       elements(64, 32, 3, 3)) * sizeof(float);
            if (!memory_available(bytes * 4)) {
                continue;
            }
            Tensor input = filled({1, 32, half, half}, 0.1f);
            Tensor weights = filled({64, 32, 3, 3}, 0.02f);
            Tensor bias = filled({64}, 0.0f);

            const double flops = 2.0 * 64 * 32 * 9 * static_cast<double>(half) * half;
            const Stats ms = sample_ms([&] {
                Tensor result = matrix_pro::conv2d(input, weights, bias, 1, 1);
                (void)result;
            }, options);
            add_result(out, "conv", "conv2d 32->64 p1 " + std::to_string(edge), edge, ms,
                       to_gflops(flops, ms.median), "GFLOPS", backend);
        }

        // --- pooling ----------------------------------------------------------
        {
            const std::size_t bytes = elements(1, 16, edge, edge) * sizeof(float);
            if (!memory_available(bytes * 4)) {
                continue;
            }
            Tensor input = filled({1, 16, edge, edge}, 0.25f);

            const Stats max_pool = sample_ms([&] {
                Tensor result = matrix_pro::max_pool2d(input, 2, 2, 0);
                (void)result;
            }, options);
            add_result(out, "conv", "max_pool2d k2 " + std::to_string(edge), edge, max_pool,
                       to_gbps(2 * bytes, max_pool.median), "GB/s", backend);

            const Stats avg_pool = sample_ms([&] {
                Tensor result = matrix_pro::avg_pool2d(input, 2, 2, 0);
                (void)result;
            }, options);
            add_result(out, "conv", "avg_pool2d k2 " + std::to_string(edge), edge, avg_pool,
                       to_gbps(2 * bytes, avg_pool.median), "GB/s", backend);
        }
    }
}

} // namespace

const BenchmarkInfo& conv_benchmark() {
    static const BenchmarkInfo info{"conv",
                                    "NCHW conv2d / pooling: forward and backward throughput",
                                    run};
    return info;
}

} // namespace bench
} // namespace matrix_pro