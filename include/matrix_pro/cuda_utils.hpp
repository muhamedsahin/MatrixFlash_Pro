#pragma once

#include <cuda_runtime.h>

namespace matrix_pro {

void checkCuda(cudaError_t status, const char* operation);
void synchronize();

}
