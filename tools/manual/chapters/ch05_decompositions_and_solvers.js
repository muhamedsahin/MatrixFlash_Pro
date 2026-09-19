// tools/manual/chapters/ch05_decompositions_and_solvers.js

module.exports = `
  <div class="chapter">
    <div class="chapter-header">
      <div class="chapter-num">Chapter 05</div>
      <h1 class="chapter-title">Advanced Decompositions & Direct Solvers</h1>
    </div>

    <h2>5.1 LU Factorization with Partial Pivoting</h2>
    <p>
      The <strong>LU Decomposition with Partial Pivoting</strong> factors any non-singular square matrix $\mathbf{A} \in \mathbb{R}^{N \times N}$ into the product of a permutation matrix $\mathbf{P}$, a unit lower triangular matrix $\mathbf{L}$, and an upper triangular matrix $\mathbf{U}$:
    </p>

    <div class="formula-box">
      $$\mathbf{P} \mathbf{A} = \mathbf{L} \mathbf{U}$$
      <span class="eq-desc">
        Where $L_{ii} = 1$, $L_{ij} = 0$ for $i < j$, and $U_{ij} = 0$ for $i > j$. Computational complexity: $\frac{2}{3} N^3$ FLOPs.
      </span>
    </div>

    <p>
      Partial pivoting exchanges rows at each step $k$ to place the maximum absolute element on the diagonal: $|A_{p, k}| = \max_{i \ge k} |A_{i, k}|$. This bounds error propagation by ensuring all multipliers in $\mathbf{L}$ satisfy $|L_{ik}| \le 1$.
    </p>

    <p>
      MatrixFlash-Pro integrates high-performance cuSOLVER dense routines (<code>cusolverDnSgetrf</code>) alongside a custom register-tiled blocked solver for small-to-medium matrices:
    </p>

    <pre><code>// MatrixFlash-Pro LU Factorization Pipeline
std::pair&lt;Matrix, std::vector&lt;int&gt;&gt; Matrix::lu_factorize() const {
    MATRIX_ASSERT_SQUARE(*this);
    int N = static_cast&lt;int&gt;(rows());
    Matrix lu = this-&gt;clone();
    
    // Allocate device pivot array
    int* d_ipiv = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ipiv, N * sizeof(int)));
    
    // Query workspace requirement
    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnSgetrf_bufferSize(
        CudaContext::instance().cusolver_handle(), N, N, lu.data(), N, &lwork));
    
    DeviceBuffer workspace(lwork * sizeof(float));
    int* d_info = nullptr;
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

    CUSOLVER_CHECK(cusolverDnSgetrf(
        CudaContext::instance().cusolver_handle(), N, N, lu.data(), N,
        reinterpret_cast&lt;float*&gt;(workspace.get()), d_ipiv, d_info));

    // Extract pivot vector to host
    std::vector&lt;int&gt; h_ipiv(N);
    CUDA_CHECK(cudaMemcpy(h_ipiv.data(), d_ipiv, N * sizeof(int), cudaMemcpyDeviceToHost));
    
    cudaFree(d_info);
    cudaFree(d_ipiv);
    return {lu, h_ipiv};
}</code></pre>

    <div class="page-subbreak"></div>

    <h2>5.2 Cholesky Decomposition for Symmetric Positive-Definite Systems</h2>
    <p>
      When a matrix $\mathbf{A}$ is Symmetric Positive-Definite ($\mathbf{A} = \mathbf{A}^T$ and $\mathbf{x}^T \mathbf{A} \mathbf{x} > 0 \quad \forall \mathbf{x} \neq \mathbf{0}$), it admits the unique <strong>Cholesky Factorization</strong>:
    </p>

    <div class="formula-box">
      $$\mathbf{A} = \mathbf{L} \mathbf{L}^T$$
      <span class="eq-desc">
        Where $\mathbf{L}$ is lower triangular with strictly positive diagonal entries. Complexity: $\frac{1}{3} N^3$ FLOPs (2&times; faster than LU).
      </span>
    </div>

    <p>
      The diagonal and off-diagonal recurrence equations are solved via parallel column sweeps:
    </p>

    <div class="formula-box">
      $$L_{jj} = \sqrt{A_{jj} - \sum_{k=1}^{j-1} L_{jk}^2}, \quad L_{ij} = \frac{1}{L_{jj}} \left( A_{ij} - \sum_{k=1}^{j-1} L_{ik} L_{jk} \right) \quad (i > j)$$
      <span class="eq-desc">Cholesky-Banachiewicz recurrence relations</span>
    </div>

    <h2>5.3 Singular Value Decomposition (SVD)</h2>
    <p>
      The <strong>Singular Value Decomposition (SVD)</strong> factors an arbitrary rectangular matrix $\mathbf{A} \in \mathbb{R}^{M \times N}$ into:
    </p>

    <div class="formula-box">
      $$\mathbf{A} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$$
      <span class="eq-desc">
        Where $\mathbf{U} \in \mathbb{R}^{M \times M}$ and $\mathbf{V} \in \mathbb{R}^{N \times N}$ are orthogonal matrices ($\mathbf{U}^T \mathbf{U} = \mathbf{I}$, $\mathbf{V}^T \mathbf{V} = \mathbf{I}$), and $\mathbf{\Sigma} = \text{diag}(\sigma_1, \sigma_2, \dots)$ contains non-negative singular values in descending order.
      </span>
    </div>

    <p>
      MatrixFlash-Pro provides both cuSOLVER polar Jacobi/QR iterations (<code>cusolverDnSgesvd</code>) and a specialized <strong>One-Sided GPU Jacobi Iteration Kernel</strong>. In the one-sided approach, orthogonal plane rotations $J(p, q, \theta)$ are applied to column pairs:
    </p>

    <pre><code>// One-Sided Jacobi Column Pair Rotation Kernel Concept
__global__ void one_sided_jacobi_kernel(
    float* __restrict__ A, int M, int N, int p, int q, float c, float s) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid &lt; M) {
        float a_ip = A[tid * N + p];
        float a_iq = A[tid * N + q];
        A[tid * N + p] = c * a_ip - s * a_iq;
        A[tid * N + q] = s * a_ip + c * a_iq;
    }
}</code></pre>

    <h2>5.4 Matrix Exponential via Scaling & Squaring with Padé [13/13]</h2>
    <p>
      Evaluating the matrix exponential $e^{\mathbf{A}} = \sum_{k=0}^\infty \frac{\mathbf{A}^k}{k!}$ directly via Taylor series suffers from severe truncation and catastrophic cancellation for large $\|\mathbf{A}\|$. MatrixFlash-Pro implements the gold-standard <strong>Higham Scaling and Squaring Algorithm with Padé [13/13] Rational Approximant</strong>:
    </p>

    <div class="formula-box">
      $$e^{\mathbf{A}} = \left( e^{\mathbf{A} / 2^s} \right)^{2^s} \approx \left( [Q_{13}(\mathbf{A}/2^s)]^{-1} P_{13}(\mathbf{A}/2^s) \right)^{2^s}$$
      <span class="eq-desc">
        Where $s = \max(0, \lceil \log_2 (\|\mathbf{A}\|_1 / \theta_{13}) \rceil)$ scales the matrix norm such that $\|\mathbf{A} / 2^s\|_1 \le \theta_{13} \approx 5.37$
      </span>
    </div>

    <p>
      The polynomials $P_{13}(X)$ and $Q_{13}(X)$ are evaluated efficiently using Horner's method and evaluated via Triangular Solve (<code>cublasStrsm</code>), guaranteeing 24-bit floating point precision throughout physical simulations.
    </p>
  </div>
`;

