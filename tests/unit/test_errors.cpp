#include <cmath>
#include <iostream>
#include <string>

#include "matrix_pro/core/matrix.hpp"
#include "matrix_pro/ops/operations.hpp"

#include "test_support.hpp"
using matrix_pro::test::check;

using matrix_pro::Matrix;

int main() {
    try {
        // Shape mismatch is reported through both the specific and the base type.
        {
            Matrix a{{1.0f, 2.0f}};
            Matrix b{{1.0f, 2.0f}, {3.0f, 4.0f}};
            bool specific = false, base = false;
            try { (void)(a + b); } catch (const matrix_pro::ShapeMismatchError&) { specific = true; }
            try { (void)(a + b); } catch (const matrix_pro::MatrixProError&) { base = true; }
            check(specific, "shape mismatch throws ShapeMismatchError");
            check(base, "ShapeMismatchError is a MatrixProError");
        }

        // matmul dimension mismatch.
        {
            Matrix a{{1.0f, 2.0f}};
            Matrix b{{1.0f, 2.0f}};
            bool caught = false;
            try { (void)(a * b); } catch (const matrix_pro::ShapeMismatchError&) { caught = true; }
            check(caught, "matmul dimension mismatch throws ShapeMismatchError");
        }

        // Invalid argument (bad dropout probability).
        {
            Matrix a{{1.0f, 2.0f}, {3.0f, 4.0f}};
            bool caught = false;
            try { (void)matrix_pro::dropout(a, 1.5f); } catch (const matrix_pro::InvalidArgumentError&) { caught = true; }
            check(caught, "bad dropout probability throws InvalidArgumentError");
        }

        // Out-of-range index.
        {
            Matrix a{{1.0f, 2.0f}, {3.0f, 4.0f}};
            bool caught = false;
            try { (void)a.at(5, 5); } catch (const matrix_pro::OutOfRangeError&) { caught = true; }
            check(caught, "out-of-range at() throws OutOfRangeError");
        }

        // Serialization error.
        {
            Matrix a{{1.0f}};
            bool caught = false;
            try { a.save("no_such_directory_12345/model.mfmp"); } catch (const matrix_pro::IoError&) { caught = true; }
            check(caught, "save to a bad path throws IoError");
        }

        // The base class catch works and carries a message.
        {
            Matrix a{{1.0f, 2.0f}};
            Matrix b{{1.0f}, {2.0f}, {3.0f}};
            bool base = false;
            try { (void)(a + b); } catch (const matrix_pro::MatrixProError& e) { base = std::string(e.what()).size() > 0; }
            check(base, "MatrixProError base catch works and carries a message");
        }

        return matrix_pro::test::summary("ERRORS");
    } catch (const std::exception& e) {
        std::cerr << "EXCEPTION: " << e.what() << "\n";
        return 99;
    }
}