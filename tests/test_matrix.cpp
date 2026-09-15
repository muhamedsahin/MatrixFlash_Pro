#include <cmath>
#include <iostream>

#include "matrix_pro/matrix.hpp"

using matrix_pro::Matrix;

static int failures = 0;
static void check(bool ok, const char* msg) {
	if (!ok) { std::cout << "FAIL: " << msg << "\n"; ++failures; }
}

int main() {
	try {
		Matrix matrix{{1.0f, 2.0f}, {3.0f, 4.0f}};
		check(matrix.rows() == 2 && matrix.cols() == 2, "shape");
		check(std::abs(matrix.trace() - 5.0f) < 1e-3f, "trace");
		check(std::abs(matrix.determinant() - (-2.0f)) < 1e-3f, "determinant");

		Matrix inverse = matrix.inverse();
		inverse.download();
		check(std::abs(inverse.at(0, 0) - (-2.0f)) < 1e-4f, "inverse 00");
		check(std::abs(inverse.at(1, 1) - (-0.5f)) < 1e-4f, "inverse 11");
		check(std::abs(inverse.at(0, 1) - 1.0f) < 1e-4f, "inverse 01");
		check(std::abs(inverse.at(1, 0) - 1.5f) < 1e-4f, "inverse 10");

		// A * inv(A) = I
		Matrix prod = matrix * inverse;
		prod.download();
		check(std::abs(prod.at(0, 0) - 1.0f) < 1e-3f && std::abs(prod.at(1, 1) - 1.0f) < 1e-3f
			&& std::abs(prod.at(0, 1)) < 1e-3f && std::abs(prod.at(1, 0)) < 1e-3f, "A * inv(A) = I");

		std::cout << (failures == 0 ? "MATRIX TESTS PASSED\n" : "MATRIX TESTS FAILED\n");
		return failures == 0 ? 0 : 1;
	} catch (const std::exception& e) {
		std::cerr << "EXCEPTION: " << e.what() << "\n";
		return 99;
	}
}
