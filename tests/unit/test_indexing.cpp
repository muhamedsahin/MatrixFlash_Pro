// Unit: gather / scatter / embedding primitives.

#include <cmath>
#include <iostream>
#include <vector>

#include "matrix_pro/indexing/indexing.hpp"
#include "matrix_pro/ops/operations.hpp"

#include "test_support.hpp"

using matrix_pro::Matrix;
using matrix_pro::test::check;

int main() {
    try {
        matrix_pro::test::section("index_select");
        {
            Matrix input{{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}};   // 3x2
            Matrix rows(2, 1, std::vector<float>{2.0f, 0.0f});
            Matrix selected = matrix_pro::index_select(input, rows, 0);
            selected.download();
            check(selected.rows() == 2 && selected.cols() == 2, "row selection keeps the column count");
            check(std::abs(selected.at(0, 0) - 5.0f) < 1e-5f && std::abs(selected.at(1, 1) - 2.0f) < 1e-5f,
                  "axis 0 picks the requested rows in order");

            Matrix columns(1, 2, std::vector<float>{1.0f, 0.0f});
            Matrix by_column = matrix_pro::index_select(input, columns, 1);
            by_column.download();
            check(by_column.rows() == 3 && by_column.cols() == 2, "column selection keeps the row count");
            check(std::abs(by_column.at(2, 0) - 6.0f) < 1e-5f, "axis 1 picks the requested columns");

            Matrix gathered = matrix_pro::gather(input, rows, 0);
            gathered.download();
            check(gathered.at(0, 0) == selected.at(0, 0), "gather is an alias of index_select");

            bool threw = false;
            try {
                (void)matrix_pro::index_select(input, Matrix(1, 1, std::vector<float>{7.0f}), 0);
            } catch (const matrix_pro::OutOfRangeError&) {
                threw = true;
            }
            check(threw, "an out-of-range index throws OutOfRangeError");

            bool bad_axis = false;
            try {
                (void)matrix_pro::index_select(input, rows, 2);
            } catch (const matrix_pro::InvalidArgumentError&) {
                bad_axis = true;
            }
            check(bad_axis, "an invalid axis throws InvalidArgumentError");
        }

        matrix_pro::test::section("scatter_add");
        {
            // out[index[i]] += src[i]; index {0,1,0} therefore accumulates two
            // source rows into destination row 0.
            Matrix src{{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}};
            Matrix index(3, 1, std::vector<float>{0.0f, 1.0f, 0.0f});
            Matrix scattered = matrix_pro::scatter_add(src, index, 2);
            scattered.download();
            check(scattered.rows() == 2 && scattered.cols() == 2, "scatter_add uses the requested row count");
            check(std::abs(scattered.at(0, 0) - 6.0f) < 1e-5f && std::abs(scattered.at(0, 1) - 8.0f) < 1e-5f,
                  "duplicate indices accumulate with atomicAdd");
            check(std::abs(scattered.at(1, 0) - 3.0f) < 1e-5f && std::abs(scattered.at(1, 1) - 4.0f) < 1e-5f,
                  "the singly targeted row holds exactly one source row");
        }

        matrix_pro::test::section("embedding");
        {
            Matrix table{{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}};   // vocab 3 x dim 2
            Matrix indices(2, 1, std::vector<float>{2.0f, 0.0f});
            Matrix lookup = matrix_pro::embedding(table, indices);
            lookup.download();
            check(lookup.rows() == 2 && lookup.cols() == 2, "embedding returns one row per index");
            check(std::abs(lookup.at(0, 0) - 5.0f) < 1e-5f && std::abs(lookup.at(1, 1) - 2.0f) < 1e-5f,
                  "embedding looks up the requested rows");

            Matrix gradient{{1.0f, 1.0f}, {1.0f, 1.0f}};
            Matrix backward = matrix_pro::embedding_backward(gradient, indices, 3);
            backward.download();
            check(backward.rows() == 3 && backward.cols() == 2, "embedding_backward allocates the vocab shape");
            check(std::abs(backward.at(0, 0) - 1.0f) < 1e-5f && std::abs(backward.at(2, 1) - 1.0f) < 1e-5f,
                  "embedding_backward scatters the gradient into the used rows");
            check(std::abs(backward.at(1, 0)) < 1e-5f, "unused vocabulary rows stay zero");
        }

        matrix_pro::test::section("scatter");
        {
            Matrix src{{1.0f, 2.0f}};
            Matrix index(2, 1, std::vector<float>{1.0f, 4.0f});
            Matrix result = matrix_pro::scatter(src, index, 2, 3);
            result.download();
            check(result.rows() == 2 && result.cols() == 3, "scatter allocates (rows x cols)");
            check(std::abs(result.at(0, 1) - 1.0f) < 1e-5f, "scatter writes at flat index 1");
            check(std::abs(result.at(1, 1) - 2.0f) < 1e-5f, "scatter writes at flat index 4");
            check(std::abs(result.at(0, 0)) < 1e-5f, "untouched elements stay zero");
        }
    } catch (const std::exception& error) {
        return matrix_pro::test::fatal(error);
    }
    return matrix_pro::test::summary("INDEXING");
}