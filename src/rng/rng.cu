#include "matrix_pro/rng/rng.hpp"
#include "matrix_pro/core/matrix.hpp"
#include "matrix_pro/core/cuda_utils.hpp"

#include <cuda_runtime.h>
#include <atomic>
#include <cmath>
#include <cstdint>

namespace matrix_pro {
namespace {

std::atomic<unsigned long long>& global_seed() {
    static std::atomic<unsigned long long> seed{12345ULL};
    return seed;
}
std::atomic<unsigned long long>& global_counter() {
    static std::atomic<unsigned long long> counter{0ULL};
    return counter;
}

// MurmurHash3-style 64-bit finalizer: cheap, well-mixed and fully
// deterministic. Not the Philox counter generator -- see rng.hpp.
__device__ unsigned hash_mix(unsigned key, unsigned long long counter) {
    unsigned v = key ^ static_cast<unsigned>(counter) ^ static_cast<unsigned>(counter >> 32);
    v *= 0xD2511F53u;
    v ^= v >> 15;
    v *= 0xCD9E8D57u;
    v ^= v >> 13;
    return v;
}

__global__ void dropout_philox_kernel(const float* in, float* out, std::size_t n,
                                      float prob, unsigned long long seed,
                                      unsigned long long ctr) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned bits = hash_mix(static_cast<unsigned>(seed), ctr + i);
    const float u = static_cast<float>(bits) / 4294967296.0f;
    out[i] = (u < prob) ? 0.0f : in[i] / (1.0f - prob);
}

__global__ void box_muller_kernel(float* out, std::size_t n,
                                  unsigned long long seed, unsigned long long ctr) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float u1 = (static_cast<float>(hash_mix(static_cast<unsigned>(seed), ctr + 2 * i)) + 0.5f) / 4294967296.0f;
    const float u2 = (static_cast<float>(hash_mix(static_cast<unsigned>(seed ^ 0x9E3779B9u), ctr + 2 * i + 1)) + 0.5f) / 4294967296.0f;
    out[i] = sqrtf(-2.0f * logf(u1 + 1e-9f)) * cosf(6.28318530718f * u2);
}

__global__ void uniform_philox_kernel(float* out, std::size_t n, float low, float range,
                                      unsigned long long seed, unsigned long long ctr) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned bits = hash_mix(static_cast<unsigned>(seed), ctr + i);
    out[i] = low + range * (static_cast<float>(bits) / 4294967296.0f);
}

unsigned long long next_counter(std::size_t n) {
    return global_counter().fetch_add(n);
}

} // namespace

void rng_seed(unsigned long long seed) {
    global_seed().store(seed);
    global_counter().store(0ULL);
}

unsigned long long rng_seed_value() { return global_seed().load(); }

Matrix randn_gpu(std::size_t rows, std::size_t cols) {
    Matrix out(rows, cols);
    if (out.empty()) return out;
    box_muller_kernel<<<(out.size() + 255) / 256, 256, 0, compute_stream()>>>(
        out.device_data(), out.size(), global_seed().load(), next_counter(out.size() * 2));
    checkCuda(cudaGetLastError(), "box muller kernel launch");
    out.mark_host_stale();
    return out;
}

Matrix uniform_gpu(std::size_t rows, std::size_t cols, float low, float high) {
    if (high < low) throw InvalidArgumentError("uniform high must be >= low");
    Matrix out(rows, cols);
    if (out.empty()) return out;
    uniform_philox_kernel<<<(out.size() + 255) / 256, 256, 0, compute_stream()>>>(
        out.device_data(), out.size(), low, high - low,
        global_seed().load(), next_counter(out.size()));
    checkCuda(cudaGetLastError(), "uniform kernel launch");
    out.mark_host_stale();
    return out;
}

Matrix dropout_gpu(const Matrix& matrix, float probability) {
    if (probability < 0.0f || probability >= 1.0f)
        throw InvalidArgumentError("Dropout probability must be in [0, 1)");
    if (probability == 0.0f) return matrix;
    Matrix out(matrix.rows(), matrix.cols());
    dropout_philox_kernel<<<(matrix.size() + 255) / 256, 256, 0, compute_stream()>>>(
        matrix.device_data(), out.device_data(), matrix.size(),
        probability, global_seed().load(), next_counter(matrix.size()));
    checkCuda(cudaGetLastError(), "dropout philox kernel launch");
    out.mark_host_stale();
    return out;
}

} // namespace matrix_pro

