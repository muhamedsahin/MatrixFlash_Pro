// Unit: zero-copy strided views over device memory.

#include <cmath>
#include <iostream>
#include <vector>

#include "matrix_pro/view/view.hpp"
#include "matrix_pro/ops/elementwise.hpp"

#include "test_support.hpp"

using matrix_pro::Matrix;
using matrix_pro::MatrixView;
using matrix_pro::test::check;

int main() {
    try {
        matrix_pro::test::section("transpose_view");
        {
            Matrix m{{1.0f, 2.0f, 3.0f},
                     {4.0f, 5.0f, 6.0f}};   // 2x3
            MatrixView t = matrix_pro::transpose_view(m);
            check(t.rows() == 3 && t.cols() == 2, "transpose swaps rows and cols");
            check(!t.contiguous(), "a transposed view is strided, not contiguous");
            Matrix dense = matrix_pro::materialize(t);
            dense.download();
            check(std::abs(dense.at(2, 1) - 6.0f) < 1e-6f, "materialize(transpose) values are transposed");
            check(std::abs(dense.at(0, 1) - 4.0f) < 1e-6f, "element (0,1) is m(1,0)");
        }

        matrix_pro::test::section("slice_view");
        {
            Matrix m{{1.0f, 2.0f, 3.0f, 4.0f},
                     {5.0f, 6.0f, 7.0f, 8.0f},
                     {9.0f, 10.0f, 11.0f, 12.0f}};   // 3x4
            MatrixView s = matrix_pro::slice_view(m, 1, 3, 1, 3);
            check(s.rows() == 2 && s.cols() == 2, "slice keeps the requested window shape");
            Matrix dense = matrix_pro::materialize(s);
            dense.download();
            check(std::abs(dense.at(0, 0) - 6.0f) < 1e-6f && std::abs(dense.at(1, 1) - 11.0f) < 1e-6f,
                  "the slice contains exactly the window elements");
        }

        matrix_pro::test::section("reshape_view");
        {
            Matrix m{{1.0f, 2.0f, 3.0f, 4.0f}};   // 1x4 contiguous
            MatrixView r = matrix_pro::reshape_view(m, 2, 2);
            check(r.rows() == 2 && r.cols() == 2, "reshape keeps the element count");
            check(r.contiguous(), "a reshape of a contiguous matrix stays contiguous");
            Matrix dense = matrix_pro::materialize(r);
            dense.download();
            check(std::abs(dense.at(1, 0) - 3.0f) < 1e-6f, "row-major reshape preserves the order");

            matrix_pro::test::check_throws<matrix_pro::ShapeMismatchError>(
                [&] { (void)matrix_pro::reshape_view(m, 3, 2); },
                "a size-changing reshape throws ShapeMismatchError");
        }

        matrix_pro::test::section("as_strided_view");
        {
            Matrix m{{1.0f, 2.0f, 3.0f},
                     {4.0f, 5.0f, 6.0f}};
            // One row, stride 2 -> picks m(0,0) and m(0,2).
            MatrixView s = matrix_pro::as_strided_view(m, 1, 2, 0, 2, 2);
            check(s.rows() == 1 && s.cols() == 2, "as_strided keeps the requested shape");
            Matrix dense = matrix_pro::materialize(s);
            dense.download();
            check(std::abs(dense.at(0, 0) - 1.0f) < 1e-6f && std::abs(dense.at(0, 1) - 3.0f) < 1e-6f,
                  "custom strides pick the strided elements");

            matrix_pro::test::check_throws<matrix_pro::OutOfRangeError>(
                [&] { (void)matrix_pro::as_strided_view(m, 4, 4, 0, 1, 1); },
                "an out-of-bounds strided view throws OutOfRangeError");
        }

        matrix_pro::test::section("view_math");
        {
            Matrix a{{1.0f, 2.0f},
                     {3.0f, 4.0f}};
            Matrix b{{10.0f, 20.0f},
                     {30.0f, 40.0f}};
            MatrixView va = matrix_pro::as_strided_view(a, 2, 2, 0, 2, 1);
            MatrixView vb = matrix_pro::as_strided_view(b, 2, 2, 0, 2, 1);
            Matrix sum = matrix_pro::add_views(va, vb);
            sum.download();
            check(std::abs(sum.at(1, 1) - 44.0f) < 1e-6f, "add_views adds elementwise");

            Matrix product = matrix_pro::multiply_views(va, vb);
            product.download();
            check(std::abs(product.at(1, 0) - 90.0f) < 1e-6f, "multiply_views multiplies elementwise");
        }

        matrix_pro::test::section("copy_view_to");
        {
            Matrix m{{1.0f, 2.0f, 3.0f},
                     {4.0f, 5.0f, 6.0f}};
            MatrixView t = matrix_pro::transpose_view(m);   // 3x2 strided
            Matrix dst(3, 2);
            matrix_pro::copy_view_to(t, dst);
            dst.download();
            check(std::abs(dst.at(2, 1) - 6.0f) < 1e-6f, "copy_view_to materializes into the destination");
            check(std::abs(dst.at(0, 1) - 4.0f) < 1e-6f, "destination values match the view");
        }
    } catch (const std::exception& error) {
        return matrix_pro::test::fatal(error);
    }
    return matrix_pro::test::summary("VIEW");
}
