#include <chrono>
#include <cstdlib>
#include <iostream>

#include "matrix_pro/matrix.hpp"

int main(int argc, char** argv) {
	const std::size_t size = argc > 1 ? static_cast<std::size_t>(std::atoi(argv[1])) : 1024;
	using clock = std::chrono::high_resolution_clock;
	matrix_pro::Matrix left = matrix_pro::Matrix::ones(size, size);
	matrix_pro::Matrix right = matrix_pro::Matrix::ones(size, size);

	const auto start = clock::now();
	matrix_pro::Matrix result = left * right;
	result.download();
	const double elapsed = std::chrono::duration<double, std::milli>(clock::now() - start).count();
	const double operations = 2.0 * static_cast<double>(size) * size * size;

	std::cout << "size=" << size << " elapsed_ms=" << elapsed
			  << " gflops=" << operations / (elapsed * 1.0e6) << '\n';
	return result.at(0, 0) > 0.0f ? 0 : 1;
}
