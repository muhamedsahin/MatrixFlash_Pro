// Dahili uygulama parçası: src/rng/rng.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Mevcut hash-mix counter RNG, Box-Muller, uniform ve dropout.

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

__global__ void dropout_philox_kernel(const float* in, float* out, std::size_t n, float prob,
                                      unsigned long long seed, unsigned long long ctr) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const unsigned bits = hash_mix(static_cast<unsigned>(seed), ctr + i);
    const float u = static_cast<float>(bits) / 4294967296.0f;
    out[i] = (u < prob) ? 0.0f : in[i] / (1.0f - prob);
}

__global__ void box_muller_kernel(float* out, std::size_t n, unsigned long long seed,
                                  unsigned long long ctr) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const float u1 =
        (static_cast<float>(hash_mix(static_cast<unsigned>(seed), ctr + 2 * i)) + 0.5f) /
        4294967296.0f;
    const float u2 =
        (static_cast<float>(hash_mix(static_cast<unsigned>(seed ^ 0x9E3779B9u), ctr + 2 * i + 1)) +
         0.5f) /
        4294967296.0f;
    out[i] = sqrtf(-2.0f * logf(u1 + 1e-9f)) * cosf(6.28318530718f * u2);
}

__global__ void uniform_philox_kernel(float* out, std::size_t n, float low, float range,
                                      unsigned long long seed, unsigned long long ctr) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const unsigned bits = hash_mix(static_cast<unsigned>(seed), ctr + i);
    out[i] = low + range * (static_cast<float>(bits) / 4294967296.0f);
}
