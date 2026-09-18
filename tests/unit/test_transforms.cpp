// Unit: whole-matrix transforms (transpose / relu / softmax / flatten / slice)
// and their Matrix convenience-method counterparts.

#include <cmath>
#include <iostream>
#include <vector>

#include "matrix_pro/ops/operations.hpp"

#include "test_support.hpp"

using matrix_pro::Matrix;
using matrix_pro::test::check;

int main() {
    try {
        matrix_pro::test::section("transpose");
        {
            Matrix input{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};   // 2x3
            Matrix transposed = matrix_pro::transpose(input);
            transposed.download();
            check(transposed.rows() == 3 && transposed.cols() == 2, "transpose swaps the shape");
            check(std::abs(transposed.at(0, 1) - 4.0f) < 1e-5f &&
                      std::abs(transposed.at(2, 0) - 3.0f) < 1e-5f,
                  "transpose moves (r,c) to (c,r)");

            Matrix transposed_twice = matrix_pro::transpose(transposed);
            transposed_twice.download();
            check(std::abs(transposed_twice.at(1, 2) - 6.0f) < 1e-5f,
                  "transposing twice restores the original layout");

            Matrix method = input.transpose();
            method.download();
            check(method.at(1, 0) == transposed.at(1, 0), "Matrix::transpose matches the free function");
        }

        matrix_pro::test::section("relu");
        {
            Matrix input{{-2.0f, 0.0f, 3.0f}};
            Matrix activated = matrix_pro::relu(input);
            activated.download();
            check(activated.at(0, 0) == 0.0f && activated.at(0, 1) == 0.0f && activated.at(0, 2) == 3.0f,
                  "relu clamps negatives to zero");
            Matrix method = input.relu();
            method.download();
            check(method.at(0, 2) == activated.at(0, 2), "Matrix::relu matches the free function");
        }

        matrix_pro::test::section("softmax");
        {
            Matrix input{{1.0f, 2.0f, 3.0f}, {-1.0f, 0.0f, 1.0f}};
            Matrix softmax = matrix_pro::softmax(input);
            softmax.download();
            const float row0 = softmax.at(0, 0) + softmax.at(0, 1) + softmax.at(0, 2);
            const float row1 = softmax.at(1, 0) + softmax.at(1, 1) + softmax.at(1, 2);
            check(std::abs(row0 - 1.0f) < 1e-4f, "softmax rows sum to one");
            check(std::abs(row1 - 1.0f) < 1e-4f, "softmax rows sum to one (second row)");
            check(softmax.at(0, 2) > softmax.at(0, 0), "softmax is monotonic in the input");

            // Numerical stability: exp(1000) would overflow without the max shift.
            Matrix extreme{{1000.0f, 1000.0f, -1000.0f}};
            Matrix stable = matrix_pro::softmax(extreme);
            stable.download();
            check(std::isfinite(stable.at(0, 0)) && std::abs(stable.at(0, 0) - 0.5f) < 1e-4f,
                  "softmax stays stable for extreme inputs");
            check(stable.at(0, 2) == 0.0f, "softmax underflows gracefully for very small inputs");
        }

        matrix_pro::test::section("flatten");
        {
            Matrix input{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
            Matrix flat = matrix_pro::flatten(input);
            flat.download();
            check(flat.rows() == input.size() && flat.cols() == 1, "flatten produces a column vector");
            check(std::abs(flat.at(4, 0) - 5.0f) < 1e-5f, "flatten keeps the row-major order");
        }

        matrix_pro::test::section("slice");
        {
            Matrix input{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f}};
            Matrix patch = matrix_pro::slice(input, 1, 3, 1, 3);
            patch.download();
            check(patch.rows() == 2 && patch.cols() == 2, "slice shape follows the requested window");
            check(std::abs(patch.at(0, 0) - 5.0f) < 1e-5f && std::abs(patch.at(1, 1) - 9.0f) < 1e-5f,
                  "slice copies the selected window");

            Matrix method = input.slice(0, 1, 0, 2);
            method.download();
            check(std::abs(method.at(0, 1) - 2.0f) < 1e-5f, "Matrix::slice matches the free function");

            bool threw = false;
            try {
                (void)matrix_pro::slice(input, 0, 5, 0, 1);
            } catch (const matrix_pro::OutOfRangeError&) {
                threw = true;
            }
            check(threw, "an out-of-bounds slice throws OutOfRangeError");
        }
    } catch (const std::exception& error) {
        return matrix_pro::test::fatal(error);
    }
    return matrix_pro::test::summary("TRANSFORMS");
}