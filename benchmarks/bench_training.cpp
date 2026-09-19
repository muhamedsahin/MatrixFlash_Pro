// Training-shaped workloads.
//
//   "mlp step (h=...)"   full forward + backward through the autograd engine for
//                        a 2-layer MLP, reported as samples/second
//   "fused bias+gelu"    fused_bias_gelu(x, b) against the unfused chain
//                        gelu(broadcast_add(x, b)), with the speedup in the note
//
// The `size` CLI value is the hidden width of the MLP.

#include <cmath>
#include <cstddef>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "benchmark_support.hpp"
#include "matrix_pro/autograd/autograd.hpp"
#include "matrix_pro/core/matrix.hpp"
#include "matrix_pro/nn/fused.hpp"
#include "matrix_pro/ops/broadcast.hpp"
#include "matrix_pro/ops/elementwise.hpp"
#include "matrix_pro/ops/product.hpp"
#include "matrix_pro/ops/shape.hpp"

namespace matrix_pro {
namespace bench {
namespace {

constexpr std::size_t kBatch = 64;
constexpr std::size_t kInputs = 256;
constexpr std::size_t kClasses = 10;

std::vector<std::size_t> default_sizes() {
    return {128, 256, 512};
}

// One-hot labels for the whole batch, built on the device.
Matrix make_targets(std::size_t batch, std::size_t classes) {
    std::vector<float> labels(batch);
    for (std::size_t i = 0; i < batch; ++i) {
        labels[i] = static_cast<float>(i % classes);
    }
    return matrix_pro::one_hot(Matrix(batch, 1, labels), classes);
}

void run(const Options& options_orig, std::vector<Result>& out) {
    Options options = options_orig;
    if (options.sizes.empty()) {
        options.sizes = default_sizes();
    }

    const Matrix targets = make_targets(kBatch, kClasses);

    for (const std::size_t hidden : options.sizes) {
        if (hidden < 4) {
            continue;
        }
        if (!memory_available(8 * (hidden * kInputs + kBatch * hidden) * sizeof(float))) {
            if (!options.quiet) {
                std::cout << "  (skipping MLP h=" << hidden << ": not enough free device memory)\n";
            }
            continue;
        }

        // --- forward + backward -------------------------------------------------
        {
            Variable input(Matrix::uniform(kBatch, kInputs, -0.5f, 0.5f), false);
            Variable target(targets, false);
            // Weight shapes follow the (rows x features) * (features x units)
            // convention of Variable::matmul.
            Variable first_weights(Matrix::glorot(kInputs, hidden));
            Variable first_bias(Matrix::zeros(1, hidden));
            Variable second_weights(Matrix::glorot(hidden, kClasses));
            Variable second_bias(Matrix::zeros(1, kClasses));

            const std::function<void()> step = [&] {
                first_weights.zero_grad();
                first_bias.zero_grad();
                second_weights.zero_grad();
                second_bias.zero_grad();

                Variable hidden_layer = input.matmul(first_weights).broadcast_add(first_bias).relu();
                Variable logits = hidden_layer.matmul(second_weights).broadcast_add(second_bias);
                Variable loss = softmax_cross_entropy_loss(logits, target);
                loss.backward();
            };

            // Validation outside the timed region: the loss must be finite and
            // the weight gradients must have been produced.
            step();
            Matrix check = second_weights.grad();
            check.download();
            for (const float value : check.data()) {
                if (!std::isfinite(value)) {
                    throw std::runtime_error("MLP backward produced a non-finite gradient");
                }
            }

            const Stats ms = sample_ms(step, options);
            const double samples_per_second =
                ms.median > 0.0 ? static_cast<double>(kBatch) * 1000.0 / ms.median : 0.0;
            add_result(out, "training", "mlp step h=" + std::to_string(hidden), hidden, ms,
                       samples_per_second, "samples/s",
                       "forward + backward, batch=" + std::to_string(kBatch));
        }

        // --- fused bias + gelu vs the unfused chain -----------------------------
        {
            const std::size_t rows = kBatch * 8;
            if (!memory_available(6 * rows * hidden * sizeof(float))) {
                continue;
            }
            Matrix input = Matrix::uniform(rows, hidden, -1.0f, 1.0f);
            // fused_binary-based chains require identical operand shapes, so the
            // bias here is elementwise rather than broadcast.
            Matrix bias = Matrix::uniform(rows, hidden, -0.5f, 0.5f);
            const std::size_t bytes = 3 * rows * hidden * sizeof(float);

            const Stats fused = sample_ms([&] {
                Matrix value = matrix_pro::fused_bias_gelu(input, bias);
                (void)value;
            }, options);

            const Stats chained = sample_ms([&] {
                Matrix shifted = matrix_pro::broadcast_add(input, bias);
                Matrix value = matrix_pro::gelu(shifted);
                (void)value;
            }, options);

            const double speedup = fused.median > 0.0 ? chained.median / fused.median : 0.0;
            const long long tenths = static_cast<long long>(speedup * 10.0 + 0.5);
            add_result(out, "training", "fused bias+gelu h=" + std::to_string(hidden), hidden,
                       fused, to_gbps(bytes, fused.median), "GB/s",
                       "fused, " + std::to_string(tenths / 10) + "." +
                           std::to_string(tenths % 10) + "x vs chain");
            add_result(out, "training", "chain add+gelu h=" + std::to_string(hidden), hidden,
                       chained, to_gbps(5 * rows * hidden * sizeof(float), chained.median),
                       "GB/s", "unfused chain");
        }

        // --- fused GEMM + bias + relu vs multiply + add_row + relu --------------
        {
            const std::size_t m = kBatch;
            const std::size_t k = kInputs;
            const std::size_t n = hidden;
            if (!memory_available(8 * (m * k + k * n + m * n) * sizeof(float))) {
                continue;
            }
            Matrix left = Matrix::uniform(m, k, -0.5f, 0.5f);
            Matrix right = Matrix::uniform(k, n, -0.5f, 0.5f);
            Matrix bias = Matrix::uniform(1, n, -0.1f, 0.1f);
            const double flops = 2.0 * static_cast<double>(m) * n * k;

            const Stats fused = sample_ms([&] {
                Matrix value = matrix_pro::gemm_bias_relu(left, right, bias);
                (void)value;
            }, options);

            const Stats chained = sample_ms([&] {
                Matrix product = left * right;
                Matrix biased = matrix_pro::add_row_vector(product, bias);
                Matrix value = biased.relu();
                (void)value;
            }, options);

            const double speedup = fused.median > 0.0 ? chained.median / fused.median : 0.0;
            const long long tenths = static_cast<long long>(speedup * 10.0 + 0.5);
            add_result(out, "training", "fused gemm+bias+relu h=" + std::to_string(hidden),
                       hidden, fused, to_gflops(flops, fused.median), "GFLOPS",
                       "cublasLt epilogue, " + std::to_string(tenths / 10) + "." +
                           std::to_string(tenths % 10) + "x vs chain");
            add_result(out, "training", "chain gemm+bias+relu h=" + std::to_string(hidden),
                       hidden, chained, to_gflops(flops, chained.median), "GFLOPS",
                       "3 kernels");
        }
    }
}

} // namespace

const BenchmarkInfo& training_benchmark() {
    static const BenchmarkInfo info{"training",
                                    "autograd MLP step + fused activation chains",
                                    run};
    return info;
}

} // namespace bench
} // namespace matrix_pro