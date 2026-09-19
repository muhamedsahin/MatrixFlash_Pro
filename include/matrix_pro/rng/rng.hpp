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

// Extended distributions
Matrix bernoulli_gpu(std::size_t rows, std::size_t cols, float prob);
Matrix exponential_gpu(std::size_t rows, std::size_t cols, float lambda = 1.0f);
Matrix poisson_gpu(std::size_t rows, std::size_t cols, float lambda);
Matrix cauchy_gpu(std::size_t rows, std::size_t cols, float loc = 0.0f, float scale = 1.0f);
Matrix log_normal_gpu(std::size_t rows, std::size_t cols, float mean = 0.0f, float stddev = 1.0f);
Matrix truncated_normal_gpu(std::size_t rows, std::size_t cols, float low = -2.0f, float high = 2.0f);
Matrix randint_gpu(std::size_t rows, std::size_t cols, int low, int high);
Matrix randperm_gpu(std::size_t n);

// Generator object with isolated state
class RNGState {
public:
    explicit RNGState(unsigned long long seed = 42);
    void manual_seed(unsigned long long seed);
    unsigned long long seed() const noexcept;
    Matrix randn(std::size_t rows, std::size_t cols);
    Matrix uniform(std::size_t rows, std::size_t cols, float low = 0.0f, float high = 1.0f);
    Matrix bernoulli(std::size_t rows, std::size_t cols, float prob);
private:
    unsigned long long seed_;
    unsigned long long counter_;
};

} // namespace matrix_pro
