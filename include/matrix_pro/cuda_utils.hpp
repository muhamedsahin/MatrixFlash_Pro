#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace matrix_pro {

void* allocate_device_memory(std::size_t bytes);
void free_device_memory(void* pointer);
void checkCuda(cudaError_t status, const char* operation);
void synchronize();

}
