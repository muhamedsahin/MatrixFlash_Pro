// tools/manual/chapters/ch09_sparse_engine.js

module.exports = `
  <div class="chapter">
    <div class="chapter-header">
      <div class="chapter-num">Chapter 09</div>
      <h1 class="chapter-title">Sparse Matrix Engine: Storage Schemes & SpMM / SpGEMM</h1>
    </div>

    <h2>9.1 Sparse Storage Formats & Memory Complexity</h2>
    <p>
      In graph neural networks, finite element analysis, and natural language adjacency matrices, matrices are typically <strong>sparse</strong> ($>99\%$ zero elements). Storing these matrices densely in VRAM is computationally infeasible and exceeds physical GPU memory capacities. MatrixFlash-Pro implements three primary sparse storage formats:
    </p>

    <table>
      <thead>
        <tr>
          <th>Format</th>
          <th>Internal Representation Arrays</th>
          <th>Memory Footprint (Bytes)</th>
          <th>Best Use Case</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><strong>COO</strong> (Coordinate)</td>
          <td><code>row_indices[nnz]</code>, <code>col_indices[nnz]</code>, <code>values[nnz]</code></td>
          <td>$12 \times \text{nnz}$</td>
          <td>Dynamic insertion, graph edge list ingestion</td>
        </tr>
        <tr>
          <td><strong>CSR</strong> (Compressed Row)</td>
          <td><code>row_ptr[M + 1]</code>, <code>col_indices[nnz]</code>, <code>values[nnz]</code></td>
          <td>$4(M + 1) + 8 \times \text{nnz}$</td>
          <td>Row-wise slicing, fast SpMM matrix multiply</td>
        </tr>
        <tr>
          <td><strong>CSC</strong> (Compressed Col)</td>
          <td><code>col_ptr[N + 1]</code>, <code>row_indices[nnz]</code>, <code>values[nnz]</code></td>
          <td>$4(N + 1) + 8 \times \text{nnz}$</td>
          <td>Column slicing, transposed operations</td>
        </tr>
      </tbody>
    </table>

    <div class="formula-box">
      $$\text{Compression Factor} = \frac{4 M N}{4(M + 1) + 8 \times \text{nnz}} \approx \frac{M N}{2 \times \text{nnz}} \quad (\text{for large } M)$$
      <span class="eq-desc">For $100,000 \times 100,000$ matrix with $0.01\%$ non-zeros, CSR reduces memory from 40 GB to 12.4 MB (3,200&times; reduction)</span>
    </div>

    <div class="page-subbreak"></div>

    <h2>9.2 Sparse-Dense Matrix Multiplication (SpMM)</h2>
    <p>
      The <strong>SpMM</strong> kernel multiplies a sparse matrix $\mathbf{A} \in \mathbb{R}^{M \times K}$ in CSR format with a dense matrix $\mathbf{B} \in \mathbb{R}^{K \times N}$, yielding a dense output $\mathbf{C} \in \mathbb{R}^{M \times N}$:
    </p>

    <div class="formula-box">
      $$\mathbf{C}_{i, j} = \sum_{k \in \text{row}(i)} \mathbf{A}_{i, k} \cdot \mathbf{B}_{k, j}$$
      <span class="eq-desc">Inner product evaluated only across non-zero elements of row $i$</span>
    </div>

    <p>
      Simple scalar kernels assign one thread per row, which causes extreme <strong>Warp Divergence</strong> when row degree distributions are skewed (power-law graphs). MatrixFlash-Pro solves this with an optimized <strong>Warp-Per-Row CSR SpMM Kernel</strong>:
    </p>

    <pre><code>// High-Throughput Warp-Per-Row CSR SpMM Kernel
__global__ void spmm_csr_warp_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_indices,
    const float* __restrict__ values,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N) {
    
    // Each warp (32 threads) processes one sparse row collaboratively
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;
    int row = warp_id;

    if (row &gt;= M) return;

    int row_start = row_ptr[row];
    int row_end = row_ptr[row + 1];

    // Stride across dense columns N
    for (int col = 0; col &lt; N; ++col) {
        float sum = 0.0f;
        // Collaborative unrolled non-zero reduction across 32 warp lanes
        for (int idx = row_start + lane_id; idx &lt; row_end; idx += 32) {
            int k = col_indices[idx];
            float val = values[idx];
            sum = fmaf(val, B[k * N + col], sum);
        }

        // Warp shuffle tree reduction across 32 lanes
        #pragma unroll
        for (int offset = 16; offset &gt; 0; offset /= 2) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }

        if (lane_id == 0) {
            C[row * N + col] = sum;
        }
    }
}</code></pre>

    <h2>9.3 Sparse-Sparse Multiplication (SpGEMM) via Gustavson's Algorithm</h2>
    <p>
      When multiplying two sparse matrices $\mathbf{C} = \mathbf{A}_{\text{sparse}} \times \mathbf{B}_{\text{sparse}}$, the output $\mathbf{C}$ is also sparse. Determining the sparsity pattern of $\mathbf{C}$ requires a two-phase symbolic and numeric algorithm:
    </p>
    <ul>
      <li><strong>Phase 1 (Symbolic Factorization):</strong> Counts the exact number of non-zero entries per output row without computing numerical products, allocating the CSR structure on the GPU.</li>
      <li><strong>Phase 2 (Numeric Accumulation):</strong> Employs a dense accumulator staged in fast Shared Memory (hash table or dense row buffer) to accumulate products before scattering to final CSR storage.</li>
    </ul>
  </div>
`;

