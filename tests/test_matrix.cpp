#include <cassert>
#include <cmath>

#include "matrix_pro/matrix.hpp"

int main() {
	using matrix_pro::Matrix;
	Matrix matrix{{1.0f, 2.0f}, {3.0f, 4.0f}};
	assert(matrix.rows() == 2 && matrix.cols() == 2);
	assert(matrix.trace() == 5.0f);
	assert(matrix.determinant() == -2.0f);

	Matrix inverse = matrix.inverse();
	inverse.download();
	assert(std::abs(inverse.at(0, 0) - (-2.0f)) < 1e-5f);
	assert(std::abs(inverse.at(1, 1) - (-0.5f)) < 1e-5f);
	return 0;
}
