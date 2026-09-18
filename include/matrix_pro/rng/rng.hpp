#pragma once

// Device-side RNG backend: deterministic, counter-based generation.
//
// Each element derives its random bits from hash_mix(seed ^ stream_counter),
// a MurmurHash3-style finalizer (64-bit multiply + xorshift mixing) evaluated
// directly inside the kernels in operations_rng.cu. cuRAND is intentionally
// NOT used: a pure counter-based scheme keeps every factory reproducible and
// allocation-free. randn uses Box-Muller over two counters per element.

#include <cstddef>
#include <cstdint>

namespace matrix_pro {

class Matrix;

// Global seed control for the device RNG backend.
void rng_seed(unsigned long long seed);
unsigned long long rng_seed_value();

// Device-side factories (GPU generation, no host round-trip).
Matrix randn_gpu(std::size_t rows, std::size_t cols);
Matrix uniform_gpu(std::size_t rows, std::size_t cols, float low, float high);
// Inverted dropout with a device-generated mask (single kernel, no host traffic).
Matrix dropout_gpu(const Matrix& matrix, float probability);

} // namespace matrix_pro
