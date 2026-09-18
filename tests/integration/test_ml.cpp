// Integration: training-oriented ops (activations, normalization, dropout,
// fused losses) and the small CNN path.

#include <cmath>
#include <iostream>
#include <vector>

#include "matrix_pro/autograd/autograd.hpp"
#include "matrix_pro/ops/operations.hpp"

#include "test_support.hpp"

using matrix_pro::Matrix;
using matrix_pro::Tensor;
using matrix_pro::Variable;
using matrix_pro::test::check;

namespace {

bool all_finite(const Matrix& matrix) {
    Matrix copy = matrix;
    copy.download();
    for (const float value : copy.data()) {
        if (!std::isfinite(value)) return false;
    }
    return true;
}

} // namespace

int main() {
    try {
        // --- activation chains --------------------------------------------------
        matrix_pro::test::section("activation chains");
        {
            Matrix input{{-1.0f, 0.0f, 1.0f}, {2.0f, -2.0f, 0.5f}};
            Matrix activation = input.leaky_relu().gelu().swish();
            check(all_finite(activation), "leaky_relu -> gelu -> swish stays finite");

            check(all_finite(matrix_pro::mish(input)), "mish is finite");
            check(all_finite(matrix_pro::softplus(input)), "softplus is finite");
            check(all_finite(matrix_pro::selu(input)), "selu is finite");
            check(all_finite(matrix_pro::hardswish(input)), "hardswish is finite");

            Matrix alpha{{0.25f, 0.25f, 0.25f}};
            Matrix prelu = matrix_pro::prelu(input, alpha);
            prelu.download();
            check(std::abs(prelu.at(0, 0) - (-0.25f)) < 1e-5f, "prelu scales negatives by alpha");
        }

        // --- normalization -----------------------------------------------------
        matrix_pro::test::section("normalization");
        {
            Matrix input{{-1.0f, 0.0f, 1.0f}, {2.0f, -2.0f, 0.5f}};
            Matrix normalized = matrix_pro::layer_norm(input);
            normalized.download();
            check(std::abs(normalized.at(0, 0) + normalized.at(0, 1) + normalized.at(0, 2)) < 1e-3f,
                  "layer_norm rows have zero mean");

            Matrix batched = matrix_pro::batch_norm(input);
            batched.download();
            const float column_mean = (batched.at(0, 0) + batched.at(1, 0)) * 0.5f;
            check(std::abs(column_mean) < 1e-3f, "batch_norm columns have zero mean");
        }

        // --- dropout -----------------------------------------------------------
        matrix_pro::test::section("dropout");
        {
            Matrix dropped = matrix_pro::dropout(Matrix::ones(2, 32), 0.25f, 42);
            check(dropped.any() && !dropped.all(), "dropout keeps part of the input");
            Matrix same_seed = matrix_pro::dropout(Matrix::ones(2, 32), 0.25f, 42);
            Matrix first = dropped;
            Matrix second = same_seed;
            first.download();
            second.download();
            check(first.data() == second.data(), "dropout is reproducible for a fixed seed");
        }

        // --- fused losses -------------------------------------------------------
        matrix_pro::test::section("fused losses");
        {
            Matrix logits{{2.0f, -1.0f, 0.5f}, {0.1f, 0.2f, 3.0f}};
            Matrix target{{1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}};
            const float bce = matrix_pro::bce_with_logits_value(logits, target);
            check(std::isfinite(bce) && bce > 0.0f, "bce_with_logits_value is positive and finite");
            check(all_finite(matrix_pro::bce_with_logits_grad(logits, target)),
                  "bce_with_logits_grad is finite");

            Matrix per_row = matrix_pro::softmax_cross_entropy_value(logits, target);
            per_row.download();
            check(per_row.rows() == 2 && per_row.cols() == 1,
                  "cross entropy returns one loss per row");
            check(per_row.at(0, 0) > 0.0f && per_row.at(1, 0) > 0.0f,
                  "cross entropy losses are positive");
            check(all_finite(matrix_pro::softmax_cross_entropy_grad(logits, target)),
                  "softmax_cross_entropy_grad is finite");
        }

        // --- autograd end-to-end ------------------------------------------------
        matrix_pro::test::section("autograd");
        {
            Variable left(Matrix{{1.0f, 2.0f}, {3.0f, 4.0f}});
            Variable right(Matrix{{2.0f, 0.0f}, {1.0f, 2.0f}});
            Variable output = left.matmul(right).relu();
            output.backward();
            Matrix left_gradient = left.grad();
            left_gradient.download();
            check(left_gradient.rows() == 2 && left_gradient.cols() == 2,
                  "matmul gradient keeps the operand shape");
            check(left_gradient.all(), "relu upstream keeps the gradient non-negative");
        }

        // --- convolution / pooling forward --------------------------------------
        matrix_pro::test::section("convolution");
        {
            Tensor image({1, 1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
            Tensor weights({1, 1, 2, 2}, {1, 1, 1, 1});
            Tensor bias({1}, {0});
            Tensor convolution = matrix_pro::conv2d(image, weights, bias);
            convolution.download();
            check(convolution.shape() == std::vector<std::size_t>{1, 1, 2, 2}, "conv2d output shape");
            check(std::abs(convolution.data()[0] - 12.0f) < 1e-4f, "conv2d top-left window sum");

            Tensor pooled = matrix_pro::max_pool2d(image, 2, 1);
            pooled.download();
            check(pooled.shape() == std::vector<std::size_t>{1, 1, 2, 2}, "max_pool2d output shape");
            check(std::abs(pooled.data()[0] - 5.0f) < 1e-4f, "max_pool2d picks the maximum");

            Tensor averaged = matrix_pro::avg_pool2d(image, 2, 1);
            averaged.download();
            check(std::abs(averaged.data()[0] - 3.0f) < 1e-4f, "avg_pool2d averages the window");
        }
    } catch (const std::exception& error) {
        return matrix_pro::test::fatal(error);
    }
    return matrix_pro::test::summary("ML");
}
