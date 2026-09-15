#include <cmath>
#include <iostream>
#include <vector>

#include "matrix_pro/operations.hpp"

using matrix_pro::Matrix;
using matrix_pro::Tensor;

int main() {
    try {
        Tensor left({2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
        Tensor right({2, 2, 2}, {1, 0, 0, 1, 2, 1, 1, 2});
        Tensor result = matrix_pro::batch_matmul(left, right);
        result.download();
        if (result.shape() != std::vector<std::size_t>{2, 2, 2}) return 1;
        if (std::abs(result.data()[0] - 1.0f) > 1e-4f || std::abs(result.data()[3] - 4.0f) > 1e-4f) return 2;

        Matrix a{{1.0f, 2.0f}, {3.0f, 4.0f}};
        Matrix b{{0.0f, 5.0f}, {6.0f, 7.0f}};
        Matrix k = a.kron(b);
        k.download();
        if (k.rows() != 4 || k.cols() != 4 || std::abs(k.at(2, 1) - 15.0f) > 1e-4f) return 3;
        a += b;
        a.download();
        if (std::abs(a.at(1, 1) - 11.0f) > 1e-4f) return 4;
        Matrix p = matrix_pro::matmul_half(Matrix{{1.0f, 2.0f}, {3.0f, 4.0f}}, Matrix{{2.0f, 0.0f}, {1.0f, 2.0f}});
        p.download();
        if (std::abs(p.at(1, 1) - 8.0f) > 1e-2f) return 5;
        Matrix d = matrix_pro::matmul_double_accumulate(Matrix{{1.0f, 2.0f}, {3.0f, 4.0f}}, Matrix{{2.0f, 0.0f}, {1.0f, 2.0f}});
        d.download();
        if (std::abs(d.at(1, 1) - 8.0f) > 1e-4f) return 6;
        Matrix values{{1.0f, 2.0f, 3.0f, 4.0f}};
        Matrix mask{{1.0f, 0.0f, 1.0f, 0.0f}};
        Matrix filtered = values.filter_by_mask(mask);
        filtered.download();
        if (filtered.size() != 2 || std::abs(filtered.data()[0] + filtered.data()[1] - 4.0f) > 1e-4f) return 7;
        if (!mask.any() || !mask.less(5.0f).all()) return 8;
        std::cout << "BATCH AND EXTENDED TESTS PASSED\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 99;
    }
}