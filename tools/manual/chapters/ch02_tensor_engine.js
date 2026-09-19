// tools/manual/chapters/ch02_tensor_engine.js

module.exports = `
  <div class="chapter">
    <div class="chapter-header">
      <div class="chapter-num">Chapter 02</div>
      <h1 class="chapter-title">Tensor Engine: Multi-Dimensional Strided Math</h1>
    </div>

    <h2>2.1 Mathematical Formulation of N-Dimensional Strides</h2>
    <p>
      In MatrixFlash-Pro, multidimensional tensors are represented as an abstraction layer over a linear, contiguous block of GPU device memory. An $N$-dimensional tensor is uniquely determined by a 4-tuple:
    </p>

    <div class="formula-box">
      $$\mathcal{T} = \langle \mathcal{D}_{\text{ptr}}, \text{Offset}, \mathbf{S}, \mathbf{s} \rangle$$
      <span class="eq-desc">
        Where $\mathbf{S} = [S_0, S_1, \dots, S_{D-1}]$ is the Shape tuple, and $\mathbf{s} = [s_0, s_1, \dots, s_{D-1}]$ is the Stride tuple.
      </span>
    </div>

    <p>
      Given an arbitrary coordinate vector $\mathbf{x} = [x_0, x_1, \dots, x_{D-1}]$ where $0 \le x_i < S_i$, the physical memory location of the scalar element is mapped through the inner product with the stride vector:
    </p>

    <div class="formula-box">
      $$\text{MemoryAddress}(\mathbf{x}) = \mathcal{D}_{\text{ptr}} + \left( \text{Offset} + \sum_{i=0}^{D-1} x_i \cdot s_i \right) \times \text{sizeof}(T)$$
      <span class="eq-desc">Linear Address Mapping Equation for arbitrary strided tensors</span>
    </div>

    <h2>2.2 Contiguity Analysis: Row-Major vs Column-Major</h2>
    <p>
      High-performance linear algebra libraries require specific memory orders to ensure optimal hardware vectorization. A tensor is strictly <strong>C-Contiguous (Row-Major)</strong> if and only if adjacent elements along the innermost dimension reside consecutively in memory:
    </p>

    <div class="formula-box">
      $$s_{D-1} = 1, \quad s_i = s_{i+1} \cdot S_{i+1} \quad \forall i \in \{0, 1, \dots, D-2\}$$
      <span class="eq-desc">Condition for C-Contiguity (Row-Major Ordering)</span>
    </div>

    <p>
      Conversely, <strong>Fortran-Contiguity (Column-Major)</strong> occurs when adjacent elements along the outermost dimension are contiguous ($s_0 = 1, s_i = s_{i-1} \cdot S_{i-1}$). MatrixFlash-Pro tracks contiguity flags dynamically in $O(D)$ time, allowing kernels to pick coalesced fast-paths whenever possible.
    </p>

    <pre><code>// MatrixFlash-Pro Contiguity Verification Logic
template &lt;size_t Dims&gt;
bool is_contiguous_c_order(const std::array&lt;size_t, Dims&gt;& shape,
                           const std::array&lt;int64_t, Dims&gt;& strides) noexcept {
    int64_t expected_stride = 1;
    for (int i = static_cast&lt;int&gt;(Dims) - 1; i &gt;= 0; --i) {
        if (shape[i] == 0) return true;
        if (shape[i] == 1) continue; // Size-1 dimensions don't break contiguity
        if (strides[i] != expected_stride) return false;
        expected_stride *= static_cast&lt;int64_t&gt;(shape[i]);
    }
    return true;
}</code></pre>

    <div class="page-subbreak"></div>

    <h2>2.3 Zero-Copy Tensor Transformations</h2>
    <p>
      A foundational design principle of MatrixFlash-Pro is that structural operations (slicing, transposition, permutation, and reshaping) must never allocate or copy GPU memory. Instead, they produce new <strong>Tensor Views</strong> in $O(1)$ CPU time by mathematically altering the stride and shape vectors.
    </p>

    <table>
      <thead>
        <tr>
          <th>Operation</th>
          <th>Mathematical Transformation</th>
          <th>Physical VRAM Copy?</th>
          <th>Complexity</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><code>transpose(dim0, dim1)</code></td>
          <td>$\text{swap}(S_{\text{dim0}}, S_{\text{dim1}}), \quad \text{swap}(s_{\text{dim0}}, s_{\text{dim1}})$</td>
          <td><strong>No (Zero-Copy)</strong></td>
          <td>$O(1)$</td>
        </tr>
        <tr>
          <td><code>permute(order)</code></td>
          <td>$S'_i = S_{\text{order}[i]}, \quad s'_i = s_{\text{order}[i]}$</td>
          <td><strong>No (Zero-Copy)</strong></td>
          <td>$O(D)$</td>
        </tr>
        <tr>
          <td><code>slice(dim, start, end, step)</code></td>
          <td>$\text{Offset}' = \text{Offset} + \text{start} \cdot s_{\text{dim}}, \quad s'_{\text{dim}} = s_{\text{dim}} \cdot \text{step}$</td>
          <td><strong>No (Zero-Copy)</strong></td>
          <td>$O(1)$</td>
        </tr>
        <tr>
          <td><code>view(new_shape)</code></td>
          <td>$\prod S'_i = \prod S_i \quad (\text{requires contiguous})$</td>
          <td><strong>No (Zero-Copy)</strong></td>
          <td>$O(D)$</td>
        </tr>
        <tr>
          <td><code>contiguous()</code></td>
          <td>Allocates contiguous buffer and executes Repack Kernel</td>
          <td><strong>Yes (GPU Kernel)</strong></td>
          <td>$O(N)$</td>
        </tr>
      </tbody>
    </table>

    <h2>2.4 Broadcasting Algebra with Zero-Stride Dimensions</h2>
    <p>
      Broadcasting enables element-wise operations between tensors of differing ranks or dimensions without data replication. Given shapes $\mathbf{A} \in \mathbb{R}^{M \times 1}$ and $\mathbf{B} \in \mathbb{R}^{1 \times N}$, the broadcasted output has shape $\mathbb{R}^{M \times N}$.
    </p>
    <p>
      MatrixFlash-Pro implements broadcasting on the GPU with <strong>Zero-Stride Representation</strong>: setting the stride of any dimension with size 1 to zero ($s_k = 0$). When a thread evaluates coordinates across that dimension, the memory offset advances by $x_k \cdot 0 = 0$, repeatedly referencing the same physical memory cell across all iterations with zero extra VRAM allocation!
    </p>

    <div class="formula-box">
      $$\forall x_k \in \{0, 1, \dots, S_k - 1\}: \quad x_k \cdot s_k = x_k \cdot 0 = 0$$
      <span class="eq-desc">Zero-Stride Broadcasting Identity: Infinite virtual expansion at zero memory cost</span>
    </div>

    <h2>2.5 GPU Strided-to-Contiguous Repack Kernel</h2>
    <p>
      When a tensor undergoes non-transposing reshapes after permutations, or is passed to vendor BLAS routines requiring dense column/row-major matrices, it must be materialized into a contiguous buffer. MatrixFlash-Pro provides an optimized GPU Repack Kernel:
    </p>

    <pre><code>// Optimized Multi-Dimensional Strided Copy Kernel
template &lt;int Dims&gt;
__global__ void repack_to_contiguous_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    size_t total_elements,
    const size_t* __restrict__ shape,
    const int64_t* __restrict__ src_strides,
    int64_t src_offset) {
    
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid &gt;= total_elements) return;

    // Deconstruct linear destination index into N-D coordinate
    size_t remaining = tid;
    int64_t physical_src_idx = src_offset;

    #pragma unroll
    for (int d = Dims - 1; d &gt;= 0; --d) {
        size_t coord = remaining % shape[d];
        remaining /= shape[d];
        physical_src_idx += coord * src_strides[d];
    }

    dst[tid] = src[physical_src_idx];
}</code></pre>
  </div>
`;

