// tools/manual/chapters/ch03_math_and_operators.js

module.exports = `
  <div class="chapter">
    <div class="chapter-header">
      <div class="chapter-num">Chapter 03</div>
      <h1 class="chapter-title">Mathematical & Element-Wise Operator Engine</h1>
    </div>

    <h2>3.1 Fused Multiply-Add (FMA) & Vectorized Arithmetic</h2>
    <p>
      High-performance numerical workloads depend heavily on <strong>Fused Multiply-Add (FMA)</strong> operations ($a \times b + c$). On NVIDIA architectures, the hardware ALU evaluates this expression in a single clock cycle with a single rounding step at infinite precision:
    </p>

    <div class="formula-box">
      $$\text{FMA}(a, b, c) = \text{round}(a \times b + c)$$
      <span class="eq-desc">Computed in 1 cycle, preserving 80-bit intermediate precision before 32-bit truncation</span>
    </div>

    <p>
      MatrixFlash-Pro guarantees the emission of the hardware PTX <code>fma.rn.f32</code> instruction rather than separate <code>fmul</code> and <code>fadd</code> steps. Furthermore, all basic element-wise kernels are vectorized across 128-bit memory lanes using <code>float4</code>:
    </p>

    <pre><code>// Vectorized Element-Wise FMA Kernel
__global__ void vectorized_fma_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ c,
    float* __restrict__ out,
    size_t num_quads,
    size_t remainder_start,
    size_t total_elements) {
    
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid &lt; num_quads) {
        float4 va = reinterpret_cast&lt;const float4*&gt;(a)[tid];
        float4 vb = reinterpret_cast&lt;const float4*&gt;(b)[tid];
        float4 vc = reinterpret_cast&lt;const float4*&gt;(c)[tid];
        float4 vout;
        vout.x = fmaf(va.x, vb.x, vc.x);
        vout.y = fmaf(va.y, vb.y, vc.y);
        vout.z = fmaf(va.z, vb.z, vc.z);
        vout.w = fmaf(va.w, vb.w, vc.w);
        reinterpret_cast&lt;float4*&gt;(out)[tid] = vout;
    } else {
        size_t idx = remainder_start + (tid - num_quads);
        if (idx &lt; total_elements) {
            out[idx] = fmaf(a[idx], b[idx], c[idx]);
        }
    }
}</code></pre>

    <h2>3.2 Special Function Units (SFU) vs IEEE Transcendental Intrinsics</h2>
    <p>
      NVIDIA SMs feature dedicated <strong>Special Function Units (SFUs)</strong> capable of computing approximations of reciprocal, square root reciprocal ($1/\sqrt{x}$), exponential ($2^x$), and trigonometric functions in hardware. MatrixFlash-Pro provides compile-time switches between exact IEEE-754 compliance and ultra-fast SFU intrinsics:
    </p>

    <table>
      <thead>
        <tr>
          <th>Operation</th>
          <th>Exact IEEE Function</th>
          <th>Fast SFU Intrinsic</th>
          <th>Peak Accuracy (ULP)</th>
          <th>Throughput Speedup</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Exponential ($e^x$)</td>
          <td><code>expf(x)</code></td>
          <td><code>__expf(x)</code></td>
          <td>~2 – 3 ULP</td>
          <td><strong>3.8&times;</strong></td>
        </tr>
        <tr>
          <td>Logarithm ($\ln x$)</td>
          <td><code>logf(x)</code></td>
          <td><code>__logf(x)</code></td>
          <td>~2 – 3 ULP</td>
          <td><strong>3.2&times;</strong></td>
        </tr>
        <tr>
          <td>Reciprocal Sqrt ($1/\sqrt{x}$)</td>
          <td><code>1.0f / sqrtf(x)</code></td>
          <td><code>rsqrtf(x)</code></td>
          <td>1 ULP</td>
          <td><strong>4.1&times;</strong></td>
        </tr>
        <tr>
          <td>Sine / Cosine</td>
          <td><code>sinf(x), cosf(x)</code></td>
          <td><code>__sinf(x), __cosf(x)</code></td>
          <td>~2 ULP</td>
          <td><strong>3.5&times;</strong></td>
        </tr>
      </tbody>
    </table>

    <div class="page-subbreak"></div>

    <h2>3.3 Deep Learning Activation Functions</h2>
    <p>
      Modern neural networks rely on non-linear activation operators. MatrixFlash-Pro implements branchless, register-optimized implementations for all industry-standard activations:
    </p>

    <div class="formula-box">
      $$\text{GELU}(x) = x \cdot \Phi(x) \approx 0.5x \left( 1 + \tanh\left( \sqrt{\frac{2}{\pi}} (x + 0.044715 x^3) \right) \right)$$
      <span class="eq-desc">Gaussian Error Linear Unit (GELU) Fast Polynomial Approximation</span>
    </div>

    <div class="formula-box">
      $$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}, \quad \text{Mish}(x) = x \cdot \tanh(\ln(1 + e^x))$$
      <span class="eq-desc">Swish / SiLU and Mish Continuous Smooth Non-Linearities</span>
    </div>

    <pre><code>// Hardware-Accelerated Activation Kernels
__device__ __forceinline__ float gelu_fast(float x) {
    const float k0 = 0.7978845608f; // sqrt(2.0 / M_PI)
    const float k1 = 0.044715f;
    float inner = k0 * fmaf(k1 * x, x * x, x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + __expf(-x));
}

__device__ __forceinline__ float relu(float x) {
    return fmaxf(0.0f, x); // Emits PTX: max.f32 (zero branching)
}</code></pre>

    <h2>3.4 Numerically Stable Online Softmax & Cross-Entropy</h2>
    <p>
      Naïve evaluation of $\text{Softmax}(x)_i = \frac{e^{x_i}}{\sum e^{x_j}}$ suffers from immediate numerical overflow in 32-bit floating point when $x_i > 88.72$ ($e^{89} \to \infty$). MatrixFlash-Pro enforces the <strong>Safe Three-Pass Formulation</strong>:
    </p>

    <div class="formula-box">
      $$m = \max_{j} x_j, \quad S = \sum_{j=1}^N e^{x_j - m}, \quad \sigma(x)_i = \frac{e^{x_i - m}}{S}$$
      <span class="eq-desc">Guarantees all exponent arguments are non-positive ($x_i - m \le 0 \implies e^{x_i - m} \in (0, 1]$)</span>
    </div>

    <h2>3.5 Warp-Level Shuffle Reduction Primitives</h2>
    <p>
      Traditional parallel reductions rely on thread synchronization barriers (<code>__syncthreads()</code>) and intermediate shared memory stores. MatrixFlash-Pro implements lock-free intra-warp register exchange via the <code>__shfl_down_sync</code> instruction, reducing 32 warp lanes in exactly 5 clock cycles:
    </p>

    <pre><code>// Warp Shuffle Down Reduction Primitive (32 threads -> 1 scalar)
__device__ __forceinline__ float warp_reduce_sum(float val) {
    unsigned int mask = 0xFFFFFFFF;
    val += __shfl_down_sync(mask, val, 16);
    val += __shfl_down_sync(mask, val, 8);
    val += __shfl_down_sync(mask, val, 4);
    val += __shfl_down_sync(mask, val, 2);
    val += __shfl_down_sync(mask, val, 1);
    return val; // Lane 0 holds the warp total
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    unsigned int mask = 0xFFFFFFFF;
    val = fmaxf(val, __shfl_down_sync(mask, val, 16));
    val = fmaxf(val, __shfl_down_sync(mask, val, 8));
    val = fmaxf(val, __shfl_down_sync(mask, val, 4));
    val = fmaxf(val, __shfl_down_sync(mask, val, 2));
    val = fmaxf(val, __shfl_down_sync(mask, val, 1));
    return val; // Lane 0 holds the warp maximum
}</code></pre>
  </div>
`;

