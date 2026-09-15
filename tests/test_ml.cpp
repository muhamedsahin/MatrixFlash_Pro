#include <cmath>
#include <iostream>
#include <vector>

#include "matrix_pro/autograd.hpp"
#include "matrix_pro/operations.hpp"

using matrix_pro::Matrix;
using matrix_pro::Tensor;
using matrix_pro::Variable;

int main() {
    try {
        Matrix input{{-1.0f, 0.0f, 1.0f}, {2.0f, -2.0f, 0.5f}};
        Matrix activation = input.leaky_relu().gelu().swish();
        activation.download();
        if (!std::isfinite(activation.at(0, 0))) return 1;

        Matrix normalized = matrix_pro::layer_norm(input);
        normalized.download();
        if (std::abs(normalized.at(0, 0) + normalized.at(0, 1) + normalized.at(0, 2)) > 1e-3f) return 2;

        Matrix dropped = matrix_pro::dropout(Matrix::ones(2, 32), 0.25f, 42);
        if (!dropped.any() || dropped.all()) return 3;

        Variable left(Matrix{{1.0f, 2.0f}, {3.0f, 4.0f}});
        Variable right(Matrix{{2.0f, 0.0f}, {1.0f, 2.0f}});
        Variable output = left.matmul(right).relu();
        output.backward();
        Matrix left_gradient = left.grad();
        left_gradient.download();
        if (left_gradient.rows() != 2 || left_gradient.cols() != 2 || !left_gradient.all()) return 4;

        Tensor image({1, 1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
        Tensor weights({1, 1, 2, 2}, {1, 1, 1, 1});
        Tensor bias({1}, {0});
        Tensor convolution = matrix_pro::conv2d(image, weights, bias);
        convolution.download();
        if (convolution.shape() != std::vector<std::size_t>{1, 1, 2, 2} || std::abs(convolution.data()[0] - 12.0f) > 1e-4f) return 5;
        Tensor pooled = matrix_pro::max_pool2d(image, 2, 1);
        pooled.download();
        if (pooled.shape() != std::vector<std::size_t>{1, 1, 2, 2} || std::abs(pooled.data()[0] - 5.0f) > 1e-4f) return 6;

        std::cout << "ML PHASE 7 TESTS PASSED\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 99;
    }
}
