// tools/manual/chapters/ch11_random_engine.js

module.exports = `
  <div class="chapter">
    <div class="chapter-header">
      <div class="chapter-num">Chapter 11</div>
      <h1 class="chapter-title">Pseudo-Random Number Engine: Counter-Based PRNG</h1>
    </div>

    <h2>11.1 The State-Based PRNG Bottleneck</h2>
    <p>
      Traditional pseudo-random number generators (such as Linear Congruential Generators, Mersenne Twister, or stateful cuRAND models) require each parallel thread to maintain a mutable internal state vector in memory. In massively parallel GPU architectures running $10^6$ threads simultaneously, this model incurs catastrophic drawbacks:
    </p>
    <ul>
      <li><strong>Memory Bandwidth Waste:</strong> Every generated scalar requires loading and storing 16 to 128 bytes of PRNG state from VRAM, saturating memory buses with auxiliary bookkeeping.</li>
      <li><strong>Reproducibility Fragility:</strong> Thread scheduling variations alter the sequence in which threads update shared state, destroying deterministic numerical reproducibility across runs.</li>
    </ul>

    <h2>11.2 The Counter-Based PRNG (CBPRNG) Paradigm</h2>
    <p>
      MatrixFlash-Pro implements the <strong>Philox4x32-10 Counter-Based Algorithm</strong> (Salmon et al.). In a counter-based PRNG, the $n$-th pseudo-random number is computed as a pure mathematical bijection of a static seed key $K$ and an explicit counter $C$:
    </p>

    <div class="formula-box">
      $$\mathbf{R}_n = \mathcal{F}_K(\mathbf{C}_n)$$
      <span class="eq-desc">
        Where $\mathbf{C}_n = \langle \text{global\_tid}, \text{sequence\_epoch} \rangle$ is an integer counter, requiring <strong>Zero Mutable Memory State</strong>
      </span>
    </div>

    <div class="arch-diagram">
+---------------------------------------------------------------------------------------------------+
|                        PHILOX4x32-10 COUNTER-BASED GPU RANDOM PIPELINE                            |
+---------------------------------------------------------------------------------------------------+
  Thread ID (e.g. 1048576) + Seed (e.g. 1337)
                      |
                      v
      +----------------------------------+
      |  Philox Round 1 (Feistel Bij.)   | ===> Multipliers: M0 = 0xD2511F53, M1 = 0xCD9E8D57
      |  Philox Round 2 .. Round 9       |
      |  Philox Round 10 (Final Diffuse) |
      +----------------------------------+
                      |
                      v  (Outputs 4 independent 32-bit random integers: R0, R1, R2, R3)
       +--------------+--------------+
       |                             |
       v                             v
  [ Uniform Float u in [0, 1) ]  [ Box-Muller Transform ]
  u = R * 2.3283064e-10f        z0 = sqrt(-2*ln(u1)) * cos(2*pi*u2) ===> Standard Normal N(0, 1)
    </div>

    <div class="page-subbreak"></div>

    <h2>11.3 Normal Distribution via Hardware Box-Muller Transform</h2>
    <p>
      To generate high-fidelity Gaussian random variables $\mathcal{N}(\mu, \sigma^2)$ for weight initialization and variational inference, MatrixFlash-Pro couples the Philox engine with the trigonometric <strong>Box-Muller Transform</strong>:
    </p>

    <div class="formula-box">
      $$Z_0 = \mu + \sigma \sqrt{-2 \ln U_1} \cos(2 \pi U_2), \quad Z_1 = \mu + \sigma \sqrt{-2 \ln U_1} \sin(2 \pi U_2)$$
      <span class="eq-desc">Maps two independent uniform variables $U_1, U_2 \in (0, 1)$ into two independent standard normal variables</span>
    </div>

    <pre><code>// Philox4x32-10 CUDA Device Implementation
struct Philox4x32 {
    static constexpr uint32_t M0 = 0xD2511F53u;
    static constexpr uint32_t M1 = 0xCD9E8D57u;
    static constexpr uint32_t W0 = 0x9E3779B9u;
    static constexpr uint32_t W1 = 0xBB67AE85u;

    __device__ static void round(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
                                 uint32_t k0, uint32_t k1) {
        uint64_t prod0 = static_cast&lt;uint64_t&gt;(r0) * M0;
        uint64_t prod1 = static_cast&lt;uint64_t&gt;(r2) * M1;
        uint32_t hi0 = static_cast&lt;uint32_t&gt;(prod0 &gt;&gt; 32);
        uint32_t lo0 = static_cast&lt;uint32_t&gt;(prod0);
        uint32_t hi1 = static_cast&lt;uint32_t&gt;(prod1 &gt;&gt; 32);
        uint32_t lo1 = static_cast&lt;uint32_t&gt;(prod1);

        uint32_t next_r0 = hi1 ^ r1 ^ k0;
        uint32_t next_r1 = lo1;
        uint32_t next_r2 = hi0 ^ r3 ^ k1;
        uint32_t next_r3 = lo0;
        r0 = next_r0; r1 = next_r1; r2 = next_r2; r3 = next_r3;
    }
};

__global__ void philox_normal_kernel(
    float* __restrict__ out, size_t n, uint64_t seed, uint64_t offset, float mean, float stddev) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid * 2 &gt;= n) return;

    uint32_t r0 = static_cast&lt;uint32_t&gt;(tid * 2 + offset);
    uint32_t r1 = static_cast&lt;uint32_t&gt;((tid * 2 + offset) &gt;&gt; 32);
    uint32_t r2 = 0, r3 = 0;
    uint32_t k0 = static_cast&lt;uint32_t&gt;(seed);
    uint32_t k1 = static_cast&lt;uint32_t&gt;(seed &gt;&gt; 32);

    #pragma unroll
    for (int i = 0; i &lt; 10; ++i) {
        Philox4x32::round(r0, r1, r2, r3, k0, k1);
        k0 += Philox4x32::W0;
        k1 += Philox4x32::W1;
    }

    // Convert to uniform floats in (0, 1]
    float u1 = (r0 + 1.0f) * 2.3283064e-10f;
    float u2 = (r1 + 1.0f) * 2.3283064e-10f;

    // Box-Muller evaluation via fast hardware intrinsics
    float radius = sqrtf(-2.0f * __logf(u1)) * stddev;
    float theta = 2.0f * 3.1415926535f * u2;
    float s, c;
    __sincosf(theta, &s, &c);

    out[tid * 2] = mean + radius * c;
    if (tid * 2 + 1 &lt; n) {
        out[tid * 2 + 1] = mean + radius * s;
    }
}</code></pre>
  </div>
`;

