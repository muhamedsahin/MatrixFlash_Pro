#include <cassert>
#include <cmath>

#include "matrix_pro/matrix.hpp"

int main() {
	using matrix_pro::Matrix;
	Matrix left{{1.0f, -2.0f}, {3.0f, 4.0f}};
	Matrix right{{2.0f, 1.0f}, {0.0f, 2.0f}};

	Matrix product = left * right;
	product.download();
	assert(product.at(0, 0) == 2.0f && product.at(1, 1) == 11.0f);

	Matrix activated = left.relu();
	activated.download();
	assert(activated.at(0, 1) == 0.0f && activated.at(1, 0) == 3.0f);

	Matrix probabilities = left.flatten().softmax();
	probabilities.download();
	assert(std::abs(probabilities.sum() - 1.0f) < 1e-5f);
	return 0;
}
