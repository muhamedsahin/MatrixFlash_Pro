#include <iostream>

#include "matrix_pro/matrix.hpp"

int main() {
	using matrix_pro::Matrix;

	const Matrix left{{1.0f, 2.0f}, {3.0f, 4.0f}};
	const Matrix right = Matrix::identity(2);
	Matrix result = (left * right).relu();
	result.download();
	std::cout << result.at(0, 0) << ' ' << result.at(1, 1) << '\n';
	return 0;
}
