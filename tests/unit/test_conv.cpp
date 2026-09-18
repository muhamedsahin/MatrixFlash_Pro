// Unit: NCHW convolution and pooling, forward and backward.

#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

#include "matrix_pro/nn/conv.hpp"

#include "test_support.hpp"

using matrix_pro::Tensor;
using matrix_pro::test::check;

namespace {

Tensor make_tensor(std::vector<std::size_t> shape, const std::vector<float>& values) {
    return Tensor(std::move(shape), values);
}

const float& at_nchw(const Tensor& t, std::size_t n, std::size_t c,
                     std::size_t h, std::size_t w) {
    const std::size_t C = t.shape()[1], H = t.shape()[2], W = t.shape()[3];
    return t.data()[((n * C + c) * H + h) * W + w];
}

} // namespace

int main() {
    try {
        matrix_pro::test::section("conv2d_forward");
        {
            // 1x1x2x2 input, 1x1x1x1 kernel of 1 -> output equals input.
            Tensor in2 = make_tensor({1, 1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
            Tensor k1 = make_tensor({1, 1, 1, 1}, {1.0f});
            Tensor b0 = make_tensor({1}, {0.0f});
            Tensor out = matrix_pro::conv2d(in2, k1, b0);
            out.download();
            check(out.shape() == std::vector<std::size_t>{1, 1, 2, 2}, "identity conv keeps the input shape");
            check(std::abs(at_nchw(out, 0, 0, 1, 1) - 4.0f) < 1e-5f, "an all-ones kernel reproduces the input");

            // 3x3 input, 2x2 kernel of ones -> 2x2 window sums.
            Tensor in3 = make_tensor({1, 1, 3, 3},
                                     {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
            Tensor k2 = make_tensor({1, 1, 2, 2}, {1.0f, 1.0f, 1.0f, 1.0f});
            Tensor win = matrix_pro::conv2d(in3, k2, b0);

        matrix_pro::test::section("conv2d_padding_stride");
        {
            Tensor in2 = make_tensor({1, 1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
            Tensor k1 = make_tensor({1, 1, 1, 1}, {1.0f});
            Tensor b0 = make_tensor({1}, {0.0f});
            Tensor padded = matrix_pro::conv2d(in2, k1, b0, 1, 1);
            padded.download();
            check(padded.shape() == std::vector<std::size_t>{1, 1, 4, 4},
                  "padding 1 on a 1x1 kernel grows the output by 2*pad");

            Tensor in4 = make_tensor({1, 1, 4, 4},
                                     {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                                      9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f});
            Tensor stride2 = matrix_pro::conv2d(in4, k1, b0, 2, 0);
            stride2.download();
            check(stride2.shape() == std::vector<std::size_t>{1, 1, 2, 2}, "stride 2 halves a 4x4 output");
            check(std::abs(at_nchw(stride2, 0, 0, 0, 0) - 1.0f) < 1e-5f &&
                  std::abs(at_nchw(stride2, 0, 0, 0, 1) - 3.0f) < 1e-5f &&
                  std::abs(at_nchw(stride2, 0, 0, 1, 1) - 11.0f) < 1e-5f,
                  "stride 2 samples the top-left of each window");
        }

            win.download();

        matrix_pro::test::section("pooling");
        {
            Tensor in2 = make_tensor({1, 1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
            Tensor maxp = matrix_pro::max_pool2d(in2, 2, 2, 0);
            maxp.download();
            check(maxp.shape() == std::vector<std::size_t>{1, 1, 1, 1}, "2x2 stride-2 max pool gives 1x1");
            check(std::abs(at_nchw(maxp, 0, 0, 0, 0) - 4.0f) < 1e-5f, "max pool picks the window maximum");
            Tensor avgp = matrix_pro::avg_pool2d(in2, 2, 2, 0);
            avgp.download();
            check(std::abs(at_nchw(avgp, 0, 0, 0, 0) - 2.5f) < 1e-5f, "avg pool averages the window");

        matrix_pro::test::section("backward");
        {
            // 1x1x1x1 kernel: grad flows straight through.
            Tensor in1 = make_tensor({1, 1, 1, 1}, {2.0f});
            Tensor k1 = make_tensor({1, 1, 1, 1}, {3.0f});
            Tensor b0 = make_tensor({1}, {0.0f});
            (void)matrix_pro::conv2d(in1, k1, b0);
            Tensor grad = make_tensor({1, 1, 1, 1}, {5.0f});
            Tensor d_in = matrix_pro::conv2d_input_backward(grad, in1, k1);
            d_in.download();
            check(std::abs(d_in.data()[0] - 15.0f) < 1e-4f, "input grad = grad * weight");
            Tensor d_w = matrix_pro::conv2d_weight_backward(grad, in1, k1);
            d_w.download();
            check(std::abs(d_w.data()[0] - 10.0f) < 1e-4f, "weight grad = grad * input");
            Tensor grad2 = make_tensor({1, 1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
            Tensor d_b = matrix_pro::conv2d_bias_backward(grad2);
            d_b.download();
            check(std::abs(d_b.data()[0] - 10.0f) < 1e-4f, "bias grad is the sum of the grad map");

            Tensor in2 = make_tensor({1, 1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
            Tensor gpool = make_tensor({1, 1, 1, 1}, {2.0f});
            Tensor d_max = matrix_pro::max_pool2d_backward(gpool, in2, 2, 2, 0);
            d_max.download();
            check(std::abs(d_max.data()[0]) < 1e-5f, "max-pool grad routes to the winner only");
            check(std::abs(d_max.data()[3] - 2.0f) < 1e-5f, "the winner (value 4) receives the grad");
            Tensor d_avg = matrix_pro::avg_pool2d_backward(gpool, in2, 2, 2, 0);
            d_avg.download();
            check(std::abs(d_avg.data()[0] - 0.5f) < 1e-5f && std::abs(d_avg.data()[3] - 0.5f) < 1e-5f,
                  "avg-pool grad is split evenly (grad / k^2)");
        }

        }

            check(win.shape() == std::vector<std::size_t>{1, 1, 2, 2}, "2x2 kernel shrinks the output by k-1");
            check(std::abs(at_nchw(win, 0, 0, 0, 0) - 12.0f) < 1e-4f, "top-left window sums to 12");
            check(std::abs(at_nchw(win, 0, 0, 1, 1) - 28.0f) < 1e-4f, "bottom-right window sums to 28");
            Tensor biased = matrix_pro::conv2d(in3, k2, make_tensor({1}, {10.0f}));
            biased.download();
            check(std::abs(at_nchw(biased, 0, 0, 0, 0) - 22.0f) < 1e-4f, "bias is added once per output element");
        }
    } catch (const std::exception& error) {
        return matrix_pro::test::fatal(error);
    }
    return matrix_pro::test::summary("CONV");
}
