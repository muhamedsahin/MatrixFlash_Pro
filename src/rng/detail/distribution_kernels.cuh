// Dahili uygulama parçası: src/rng/rng.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Bernoulli, exponential, Poisson ve diğer dağılım çekirdekleri.

__global__ void bernoulli_kernel(float* out, std::size_t n, float prob, unsigned long long seed,
                                 unsigned long long ctr) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const unsigned bits = hash_mix(static_cast<unsigned>(seed), ctr + i);
    const float u = static_cast<float>(bits) / 4294967296.0f;
    out[i] = (u < prob) ? 1.0f : 0.0f;
}

__global__ void exponential_kernel(float* out, std::size_t n, float lambda, unsigned long long seed,
                                   unsigned long long ctr) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const unsigned bits = hash_mix(static_cast<unsigned>(seed), ctr + i);
    float u = static_cast<float>(bits) / 4294967296.0f;
    if (u >= 1.0f - 1e-7f)
        u = 1.0f - 1e-7f;
    out[i] = -logf(1.0f - u) / lambda;
}

__global__ void poisson_kernel(float* out, std::size_t n, float lambda, unsigned long long seed,
                               unsigned long long ctr) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
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
        const float u1 =
            (static_cast<float>(hash_mix(static_cast<unsigned>(seed), ctr + 2 * i)) + 0.5f) /
            4294967296.0f;
        const float u2 = (static_cast<float>(hash_mix(static_cast<unsigned>(seed ^ 0x9E3779B9u),
                                                      ctr + 2 * i + 1)) +
                          0.5f) /
                         4294967296.0f;
        float z = sqrtf(-2.0f * logf(u1 + 1e-9f)) * cosf(6.28318530718f * u2);
        float res = roundf(lambda + sqrtf(lambda) * z);
        out[i] = (res < 0.0f) ? 0.0f : res;
    }
}

__global__ void cauchy_kernel(float* out, std::size_t n, float loc, float scale,
                              unsigned long long seed, unsigned long long ctr) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const unsigned bits = hash_mix(static_cast<unsigned>(seed), ctr + i);
    const float u = static_cast<float>(bits) / 4294967296.0f;
    out[i] = loc + scale * tanf(3.1415926535f * (u - 0.5f));
}

__global__ void log_normal_kernel(float* out, std::size_t n, float mean, float stddev,
                                  unsigned long long seed, unsigned long long ctr) {
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
    float z = sqrtf(-2.0f * logf(u1 + 1e-9f)) * cosf(6.28318530718f * u2);
    out[i] = expf(mean + stddev * z);
}

__global__ void truncated_normal_kernel(float* out, std::size_t n, float low, float high,
                                        unsigned long long seed, unsigned long long ctr) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    unsigned long long local_ctr = ctr + i;
    float val = 0.0f;
    while (true) {
        const float u1 =
            (static_cast<float>(hash_mix(static_cast<unsigned>(seed), local_ctr)) + 0.5f) /
            4294967296.0f;
        const float u2 = (static_cast<float>(
                              hash_mix(static_cast<unsigned>(seed ^ 0x9E3779B9u), local_ctr + n)) +
                          0.5f) /
                         4294967296.0f;
        val = sqrtf(-2.0f * logf(u1 + 1e-9f)) * cosf(6.28318530718f * u2);
        if (val >= low && val <= high)
            break;
        local_ctr += 2 * n;
    }
    out[i] = val;
}

__global__ void randint_kernel(float* out, std::size_t n, int low, int range,
                               unsigned long long seed, unsigned long long ctr) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const unsigned bits = hash_mix(static_cast<unsigned>(seed), ctr + i);
    out[i] = static_cast<float>(low + (bits % range));
}

__global__ void randperm_init_kernel(float* out, float* keys, std::size_t n,
                                     unsigned long long seed) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    out[i] = static_cast<float>(i);
    const unsigned bits = hash_mix(static_cast<unsigned>(seed), i);
    keys[i] = static_cast<float>(bits) / 4294967296.0f;
}
