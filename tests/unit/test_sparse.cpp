// Unit: CSR sparse-matrix construction and sparse/dense mixed operations.

#include <cmath>
#include <iostream>
#include <vector>

#include "matrix_pro/core/matrix.hpp"
#include "matrix_pro/sparse/sparse.hpp"

#include "test_support.hpp"

using matrix_pro::Matrix;
using matrix_pro::SparseCSR;
using matrix_pro::test::check;

int main() {
    try {
        matrix_pro::test::section("from_dense");
        {
            Matrix dense{{1.0f, 0.0f, 2.0f},
                         {0.0f, 0.0f, 0.0f},
                         {3.0f, 4.0f, 0.0f}};
            SparseCSR a = SparseCSR::from_dense(dense);
            check(a.rows() == 3 && a.cols() == 3, "from_dense keeps the shape");
            check(a.nnz() == 4, "from_dense drops the zeros (threshold 0)");
            check(std::abs(a.sparsity() - 5.0f / 9.0f) < 1e-6f, "sparsity equals 1 - nnz/total");

            const std::vector<int> rp = a.host_row_ptr();
            const std::vector<int> ci = a.host_col_idx();
            const std::vector<float> va = a.host_values();
            check(rp.size() == 4 && rp[0] == 0 && rp[1] == 2 && rp[2] == 2 && rp[3] == 4,
                  "row_ptr is a valid CSR prefix sum");
            check(ci[0] == 0 && ci[1] == 2 && ci[2] == 0 && ci[3] == 1,
                  "col_idx matches the non-zero pattern");
            check(std::abs(va[0] - 1.0f) < 1e-6f && std::abs(va[3] - 4.0f) < 1e-6f,
                  "values hold the non-zero entries");

            Matrix back = a.to_dense();
            back.download();
            check(std::abs(back.at(2, 1) - 4.0f) < 1e-6f, "to_dense restores the dense matrix");
            check(std::abs(back.at(1, 2)) < 1e-6f, "to_dense writes zeros where the pattern is empty");
        }

        matrix_pro::test::section("from_coo");
        {
            SparseCSR b = SparseCSR::from_coo(2, 3,
                                              {0, 0, 1},
                                              {1, 2, 0},
                                              {1.0f, 2.0f, 3.0f});
            check(b.nnz() == 3, "from_coo stores every triplet");
            Matrix dense = b.to_dense();
            dense.download();
            check(std::abs(dense.at(0, 1) - 1.0f) < 1e-6f && std::abs(dense.at(1, 0) - 3.0f) < 1e-6f,
                  "COO triplets land at (row, col)");
        }

        matrix_pro::test::section("spmv");
        {
            SparseCSR a = SparseCSR::from_coo(2, 3, {0, 1, 1}, {0, 1, 2}, {2.0f, 1.0f, 3.0f});
            Matrix x(3, 1, std::vector<float>{1.0f, 2.0f, 4.0f});
            Matrix y = matrix_pro::spmv(a, x);
            y.download();
            check(y.rows() == 2 && y.cols() == 1, "spmv output is (rows x 1)");
            check(std::abs(y.at(0, 0) - 2.0f) < 1e-5f, "row 0: 2*1");
            check(std::abs(y.at(1, 0) - 14.0f) < 1e-5f, "row 1: 1*2 + 3*4");
        }

        matrix_pro::test::section("sparse_matmul");
        {
            SparseCSR a = SparseCSR::from_coo(2, 2, {0, 1}, {1, 0}, {2.0f, 3.0f});
            Matrix b{{1.0f, 4.0f}, {5.0f, 6.0f}};
            Matrix c = matrix_pro::sparse_matmul(a, b);
            c.download();
            check(c.rows() == 2 && c.cols() == 2, "sparse_matmul output shape is (m x n)");
            check(std::abs(c.at(0, 1) - 12.0f) < 1e-5f, "row 0 = 2 * row 1 of b");
            check(std::abs(c.at(1, 0) - 3.0f) < 1e-5f, "row 1 = 3 * row 0 of b");
        }

        matrix_pro::test::section("copy_semantics");
        {
            SparseCSR original = SparseCSR::from_coo(2, 2, {0, 1}, {0, 1}, {1.0f, 2.0f});
            SparseCSR copy(original);
            check(copy.nnz() == 2 && copy.host_values()[1] == 2.0f, "deep copy keeps the values");
            SparseCSR moved = std::move(copy);
            check(moved.nnz() == 2, "move preserves the structure");
        }
    } catch (const std::exception& error) {
        return matrix_pro::test::fatal(error);
    }
    return matrix_pro::test::summary("SPARSE");
}
