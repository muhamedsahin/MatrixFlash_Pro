#include "matrix_pro/cuda_utils.hpp"

#include <stdexcept>
#include <string>

namespace matrix_pro {

void checkCuda(cudaError_t status, const char* operation) {
	if (status != cudaSuccess) {
		throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
	}
}

void synchronize() {
	checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

}
