// Integration: batched (rank-3) GEMM, mixed-precision GEMM, extended operators
// and mask-based selection.

#include <cmath>
#include <iostream>
#include <vector>

#include "matrix_pro/ops/operations.hpp"

#include "test_support.hpp"

using matrix_pro::Matrix;
using matrix_pro::Tensor;
using matrix_pro::test::check;

int main() {
    try {
        // --- batched GEMM -------------------------------------------------------
        matrix_pro::test::section("batch_matmul");
        {
            Tensor left({2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
            Tensor right({2, 2, 2}, {1, 0, 0, 1, 2, 1, 1, 2});
            Tensor result = matrix_pro::batch_matmul(left, right);
            result.download();
            check(result.shape() == std::vector<std::size_t>{2, 2, 2}, "batch_matmul keeps [batch,m,n]");
            check(std::abs(result.data()[0] - 1.0f) < 1e-4f &&
                      std::abs(result.data()[3] - 4.0f) < 1e-4f,
                  "batch 0 equals A0 * B0");

            // Gradient of the left operand is G x B^T; with G = I that is B^T.
            Tensor identity({2, 2, 2}, {1, 0, 0, 1, 1, 0, 0, 1});
            Tensor grad_left = matrix_pro::batch_matmul_backward_left(identity, right);
            grad_left.download();
            check(std::abs(grad_left.data()[1] - 0.0f) < 1e-4f &&
                      std::abs(grad_left.data()[2] - 0.0f) < 1e-4f,
                  "batch_matmul_backward_left produces B^T for G = I");

            Tensor grad_right = matrix_pro::batch_matmul_backward_right(identity, left);
            grad_right.download();
            // A0^T for G = I: [[1,3],[2,4]] -> data[1]=3, data[2]=2.
            check(std::abs(grad_right.data()[1] - 3.0f) < 1e-4f &&
                      std::abs(grad_right.data()[2] - 2.0f) < 1e-4f,
                  "batch_matmul_backward_right produces A^T for G = I");
        }

        // --- mixed precision ----------------------------------------------------
        matrix_pro::test::section("precision");
        {
            Matrix a{{1.0f, 2.0f}, {3.0f, 4.0f}};
            Matrix b{{2.0f, 0.0f}, {1.0f, 2.0f}};
            Matrix reference = a * b;

            Matrix half = matrix_pro::matmul_half(a, b);
            half.download();
            reference.download();
            check(std::abs(half.at(1, 1) - reference.at(1, 1)) < 1e-2f, "matmul_half matches fp32 closely");

            Matrix doubled = matrix_pro::matmul_double_accumulate(a, b);
            doubled.download();
            check(std::abs(doubled.at(1, 1) - 8.0f) < 1e-4f, "fp64 accumulation is numerically exact");
        }

        // --- extended operators / products -------------------------------------
        matrix_pro::test::section("extended");
        {
            Matrix a{{1.0f, 2.0f}, {3.0f, 4.0f}};
            Matrix b{{0.0f, 5.0f}, {6.0f, 7.0f}};

            Matrix k = a.kron(b);
            k.download();
            check(k.rows() == 4 && k.cols() == 4, "kron shape is (a*c x b*d)");
            check(std::abs(k.at(2, 1) - 15.0f) < 1e-4f, "kron value");

            a += b;
            a.download();
            check(std::abs(a.at(1, 1) - 11.0f) < 1e-4f, "operator+= adds elementwise");

            Matrix outer = Matrix{{1.0f, 2.0f}}.outer_product(Matrix{{3.0f, 4.0f}});
            outer.download();
            check(std::abs(outer.at(0, 0) - 3.0f) < 1e-4f && std::abs(outer.at(0, 1) - 4.0f) < 1e-4f,
                  "outer_product of two row vectors");
        }

        // --- masking / comparison ----------------------------------------------
        matrix_pro::test::section("masking");
        {
            Matrix values{{1.0f, 2.0f, 3.0f, 4.0f}};
            // Mask convention: non-zero entries are masked out (replaced).
            Matrix mask{{0.0f, 1.0f, 0.0f, 1.0f}};
            Matrix filtered = values.filter_by_mask(mask);
            filtered.download();
            check(filtered.size() == 2, "filter_by_mask keeps the selected entries");
            check(std::abs(filtered.data()[0] + filtered.data()[1] - 6.0f) < 1e-4f,
                  "filter_by_mask preserves the values");

            check(mask.any() && !mask.all(), "any/all agree with the mask contents");
            check(mask.less(5.0f).all(), "less() builds an all-true mask");

            Matrix replaced = matrix_pro::apply_mask(values, mask, -1.0f);
            replaced.download();
            check(std::abs(replaced.at(0, 1) + 1.0f) < 1e-4f, "apply_mask writes the fill value");
            check(std::abs(replaced.at(0, 2) - 3.0f) < 1e-4f, "apply_mask keeps unmasked entries");

            Matrix selected = matrix_pro::where(mask, values, Matrix::zeros(1, 4));
            selected.download();
            check(std::abs(selected.at(0, 0)) < 1e-4f && std::abs(selected.at(0, 1) - 2.0f) < 1e-4f,
                  "where() selects per element");
            Matrix inverted = matrix_pro::logical_not(mask);
            inverted.download();
            check(inverted.at(0, 0) == 1.0f && inverted.at(0, 1) == 0.0f, "logical_not inverts a mask");
            check(matrix_pro::logical_and(mask, inverted).all() == false, "a mask AND its inverse is empty");
            check(matrix_pro::logical_or(mask, inverted).all(), "a mask OR its inverse is full");
        }
    } catch (const std::exception& error) {
        return matrix_pro::test::fatal(error);
    }
    return matrix_pro::test::summary("BATCH");
}
