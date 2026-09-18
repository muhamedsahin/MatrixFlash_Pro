// ============================================================================
//  MatrixFlash-Pro — smallest end-to-end example
//  ---------------------------------------------------------------------------
//  Build and run:
//     cmake --build --preset release --target matrix_pro_basic
//     ./build/Release/matrix_pro_basic
//
//  The umbrella header below exposes every public module of the library.
//  Including the granular headers instead (e.g. <matrix_pro/ops/product.hpp>)
//  keeps compile times down in larger projects.
// ============================================================================

#include <iostream>

#include "matrix_pro/matrix_pro.hpp"

int main() {
	using matrix_pro::Matrix;

	const Matrix left{{1.0f, 2.0f}, {3.0f, 4.0f}};
	const Matrix right = Matrix::identity(2);
	Matrix result = (left * right).relu();
	result.download();
	std::cout << result.at(0, 0) << ' ' << result.at(1, 1) << '\n';
	return 0;
}
