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

namespace {
__global__ void bernoulli_kernel(float* out, std::size_t n, float prob,
                                  unsigned long long seed, unsigned long long ctr) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned bits = hash_mix(static_cast<unsigned>(seed), ctr + i);
    const float u = static_cast<float>(bits) / 4294967296.0f;
    out[i] = (u < prob) ? 1.0f : 0.0f;
}

__global__ void exponential_kernel(float* out, std::size_t n, float lambda,
                                    unsigned long long seed, unsigned long long ctr) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned bits = hash_mix(static_cast<unsigned>(seed), ctr + i);
    float u = static_cast<float>(bits) / 4294967296.0f;
    if (u >= 1.0f - 1e-7f) u = 1.0f - 1e-7f;
    out[i] = -logf(1.0f - u) / lambda;
}

__global__ void poisson_kernel(float* out, std::size_t n, float lambda,
                                unsigned long long seed, unsigned long long ctr) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (lambda <= 30.0f) {
        float L = expf(-lambda);
        float p = 1.0f;
        int k = 0;
        unsigned long long local_ctr = ctr + i;
        do {
            k++;
            const unsigned bits = hash_mix(static_cast<unsigned>(seed), local_ctr);
            local_ctr += n;
            float u = static_cast<float>(bits) / 4294967296.0f;
            p *= u;
        } while (p > L);
        out[i] = static_cast<float>(k - 1);
    } else {
        const float u1 = (static_cast<float>(hash_mix(static_cast<unsigned>(seed), ctr + 2 * i)) + 0.5f) / 4294967296.0f;
        const float u2 = (static_cast<float>(hash_mix(static_cast<unsigned>(seed ^ 0x9E3779B9u), ctr + 2 * i + 1)) + 0.5f) / 4294967296.0f;
        float z = sqrtf(-2.0f * logf(u1 + 1e-9f)) * cosf(6.28318530718f * u2);
        float res = roundf(lambda + sqrtf(lambda) * z);
        out[i] = (res < 0.0f) ? 0.0f : res;
    }
}

__global__ void cauchy_kernel(float* out, std::size_t n, float loc, float scale,
                               unsigned long long seed, unsigned long long ctr) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned bits = hash_mix(static_cast<unsigned>(seed), ctr + i);
    const float u = static_cast<float>(bits) / 4294967296.0f;
    out[i] = loc + scale * tanf(3.1415926535f * (u - 0.5f));
}

__global__ void log_normal_kernel(float* out, std::size_t n, float mean, float stddev,
                                   unsigned long long seed, unsigned long long ctr) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float u1 = (static_cast<float>(hash_mix(static_cast<unsigned>(seed), ctr + 2 * i)) + 0.5f) / 4294967296.0f;
    const float u2 = (static_cast<float>(hash_mix(static_cast<unsigned>(seed ^ 0x9E3779B9u), ctr + 2 * i + 1)) + 0.5f) / 4294967296.0f;
    float z = sqrtf(-2.0f * logf(u1 + 1e-9f)) * cosf(6.28318530718f * u2);
    out[i] = expf(mean + stddev * z);
}

__global__ void truncated_normal_kernel(float* out, std::size_t n, float low, float high,
                                         unsigned long long seed, unsigned long long ctr) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned long long local_ctr = ctr + i;
    float val = 0.0f;
    while (true) {
        const float u1 = (static_cast<float>(hash_mix(static_cast<unsigned>(seed), local_ctr)) + 0.5f) / 4294967296.0f;
        const float u2 = (static_cast<float>(hash_mix(static_cast<unsigned>(seed ^ 0x9E3779B9u), local_ctr + n)) + 0.5f) / 4294967296.0f;
        val = sqrtf(-2.0f * logf(u1 + 1e-9f)) * cosf(6.28318530718f * u2);
        if (val >= low && val <= high) break;
        local_ctr += 2 * n;
    }
    out[i] = val;
}

__global__ void randint_kernel(float* out, std::size_t n, int low, int range,
                                unsigned long long seed, unsigned long long ctr) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned bits = hash_mix(static_cast<unsigned>(seed), ctr + i);
    out[i] = static_cast<float>(low + (bits % range));
}

__global__ void randperm_init_kernel(float* out, float* keys, std::size_t n, unsigned long long seed) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = static_cast<float>(i);
    const unsigned bits = hash_mix(static_cast<unsigned>(seed), i);
    keys[i] = static_cast<float>(bits) / 4294967296.0f;
}
} // namespace

Matrix bernoulli_gpu(std::size_t rows, std::size_t cols, float prob) {
    Matrix out(rows, cols);
    if (out.empty()) return out;
    bernoulli_kernel<<<(out.size() + 255) / 256, 256, 0, compute_stream()>>>(
        out.device_data(), out.size(), prob, global_seed().load(), next_counter(out.size()));
    checkCuda(cudaGetLastError(), "bernoulli kernel launch");
    out.mark_host_stale();
    return out;
}

Matrix exponential_gpu(std::size_t rows, std::size_t cols, float lambda) {
    Matrix out(rows, cols);
    if (out.empty()) return out;
    exponential_kernel<<<(out.size() + 255) / 256, 256, 0, compute_stream()>>>(
        out.device_data(), out.size(), lambda, global_seed().load(), next_counter(out.size()));
    checkCuda(cudaGetLastError(), "exponential kernel launch");
    out.mark_host_stale();
    return out;
}

Matrix poisson_gpu(std::size_t rows, std::size_t cols, float lambda) {
    Matrix out(rows, cols);
    if (out.empty()) return out;
    unsigned long long ctr_inc = (lambda <= 30.0f) ? out.size() * static_cast<unsigned long long>(lambda + 5.0f) : out.size() * 2;
    poisson_kernel<<<(out.size() + 255) / 256, 256, 0, compute_stream()>>>(
        out.device_data(), out.size(), lambda, global_seed().load(), next_counter(ctr_inc));
    checkCuda(cudaGetLastError(), "poisson kernel launch");
    out.mark_host_stale();
    return out;
}

Matrix cauchy_gpu(std::size_t rows, std::size_t cols, float loc, float scale) {
    Matrix out(rows, cols);
    if (out.empty()) return out;
    cauchy_kernel<<<(out.size() + 255) / 256, 256, 0, compute_stream()>>>(
        out.device_data(), out.size(), loc, scale, global_seed().load(), next_counter(out.size()));
    checkCuda(cudaGetLastError(), "cauchy kernel launch");
    out.mark_host_stale();
    return out;
}

Matrix log_normal_gpu(std::size_t rows, std::size_t cols, float mean, float stddev) {
    Matrix out(rows, cols);
    if (out.empty()) return out;
    log_normal_kernel<<<(out.size() + 255) / 256, 256, 0, compute_stream()>>>(
        out.device_data(), out.size(), mean, stddev, global_seed().load(), next_counter(out.size() * 2));
    checkCuda(cudaGetLastError(), "log normal kernel launch");
    out.mark_host_stale();
    return out;
}

Matrix truncated_normal_gpu(std::size_t rows, std::size_t cols, float low, float high) {
    Matrix out(rows, cols);
    if (out.empty()) return out;
    unsigned long long ctr_inc = out.size() * 10;
    truncated_normal_kernel<<<(out.size() + 255) / 256, 256, 0, compute_stream()>>>(
        out.device_data(), out.size(), low, high, global_seed().load(), next_counter(ctr_inc));
    checkCuda(cudaGetLastError(), "truncated normal kernel launch");
    out.mark_host_stale();
    return out;
}

Matrix randint_gpu(std::size_t rows, std::size_t cols, int low, int high) {
    if (high <= low) throw InvalidArgumentError("randint high must be > low");
    Matrix out(rows, cols);
    if (out.empty()) return out;
    randint_kernel<<<(out.size() + 255) / 256, 256, 0, compute_stream()>>>(
        out.device_data(), out.size(), low, high - low, global_seed().load(), next_counter(out.size()));
    checkCuda(cudaGetLastError(), "randint kernel launch");
    out.mark_host_stale();
    return out;
}

#include <vector>
#include <algorithm>
#include <utility>

Matrix randperm_gpu(std::size_t n) {
    Matrix out(n, 1);
    if (n == 0) return out;
    Matrix keys(n, 1);
    randperm_init_kernel<<<(n + 255) / 256, 256, 0, compute_stream()>>>(
        out.device_data(), keys.device_data(), n, global_seed().load() ^ next_counter(n));
    checkCuda(cudaGetLastError(), "randperm_init_kernel launch");
    
    out.download();
    keys.download();
    std::vector<std::pair<float, float>> pairs(n);
    for (std::size_t i = 0; i < n; ++i) {
        pairs[i] = {keys.data()[i], out.data()[i]};
    }
    std::sort(pairs.begin(), pairs.end());
    for (std::size_t i = 0; i < n; ++i) {
        out.data()[i] = pairs[i].second;
    }
    out.upload();
    return out;
}

RNGState::RNGState(unsigned long long seed) : seed_(seed), counter_(0ULL) {}

void RNGState::manual_seed(unsigned long long seed) {
    seed_ = seed;
    counter_ = 0ULL;
}

unsigned long long RNGState::seed() const noexcept {
    return seed_;
}

Matrix RNGState::randn(std::size_t rows, std::size_t cols) {
    Matrix out(rows, cols);
    if (out.empty()) return out;
    box_muller_kernel<<<(out.size() + 255) / 256, 256, 0, compute_stream()>>>(
        out.device_data(), out.size(), seed_, counter_);
    counter_ += out.size() * 2;
    checkCuda(cudaGetLastError(), "box muller kernel launch");
    out.mark_host_stale();
    return out;
}

Matrix RNGState::uniform(std::size_t rows, std::size_t cols, float low, float high) {
    if (high < low) throw InvalidArgumentError("uniform high must be >= low");
    Matrix out(rows, cols);
    if (out.empty()) return out;
    uniform_philox_kernel<<<(out.size() + 255) / 256, 256, 0, compute_stream()>>>(
        out.device_data(), out.size(), low, high - low, seed_, counter_);
    counter_ += out.size();
    checkCuda(cudaGetLastError(), "uniform kernel launch");
    out.mark_host_stale();
    return out;
}

Matrix RNGState::bernoulli(std::size_t rows, std::size_t cols, float prob) {
    Matrix out(rows, cols);
    if (out.empty()) return out;
    bernoulli_kernel<<<(out.size() + 255) / 256, 256, 0, compute_stream()>>>(
        out.device_data(), out.size(), prob, seed_, counter_);
    counter_ += out.size();
    checkCuda(cudaGetLastError(), "bernoulli kernel launch");
    out.mark_host_stale();
    return out;
}

} // namespace matrix_pro

