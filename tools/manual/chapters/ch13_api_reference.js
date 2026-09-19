// tools/manual/chapters/ch13_api_reference.js

module.exports = `
  <div class="chapter">
    <div class="chapter-header">
      <div class="chapter-num">Chapter 13</div>
      <h1 class="chapter-title">Complete API Reference Directory</h1>
    </div>

    <h2>13.1 Core Matrix Interface: matrix_pro::Matrix</h2>
    <p>
      The <code>Matrix</code> class provides the primary RAII-managed GPU 2D dense matrix structure. All operations guarantee strong exception safety and automated device memory cleanup.
    </p>

    <table>
      <thead>
        <tr>
          <th>Method Signature</th>
          <th>Return Type</th>
          <th>Description & Constraints</th>
          <th>Complexity</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><code>Matrix(size_t rows, size_t cols)</code></td>
          <td>Instance</td>
          <td>Allocates uninitialized VRAM on active CUDA device.</td>
          <td>$O(1)$</td>
        </tr>
        <tr>
          <td><code>Matrix(std::initializer_list&lt;std::initializer_list&lt;float&gt;&gt;)</code></td>
          <td>Instance</td>
          <td>Constructs from host 2D literal; transparently uploads to GPU.</td>
          <td>$O(MN)$</td>
        </tr>
        <tr>
          <td><code>static Matrix zeros(size_t rows, size_t cols)</code></td>
          <td><code>Matrix</code></td>
          <td>Initializes all elements to 0.0f via fast <code>cudaMemsetAsync</code>.</td>
          <td>$O(MN)$</td>
        </tr>
        <tr>
          <td><code>static Matrix ones(size_t rows, size_t cols)</code></td>
          <td><code>Matrix</code></td>
          <td>Initializes all elements to 1.0f via vectorized fill kernel.</td>
          <td>$O(MN)$</td>
        </tr>
        <tr>
          <td><code>static Matrix identity(size_t n)</code></td>
          <td><code>Matrix</code></td>
          <td>Constructs $n \times n$ identity matrix with diagonal set to 1.0f.</td>
          <td>$O(n^2)$</td>
        </tr>
        <tr>
          <td><code>static Matrix random(size_t rows, size_t cols, float min, float max)</code></td>
          <td><code>Matrix</code></td>
          <td>Fills from Philox4x32 uniform distribution $\mathcal{U}[\text{min}, \text{max})$.</td>
          <td>$O(MN)$</td>
        </tr>
        <tr>
          <td><code>static Matrix randn(size_t rows, size_t cols, float mean, float std)</code></td>
          <td><code>Matrix</code></td>
          <td>Fills from Philox4x32 Gaussian distribution $\mathcal{N}(\text{mean}, \text{std}^2)$.</td>
          <td>$O(MN)$</td>
        </tr>
        <tr>
          <td><code>void download(float* host_dest) const</code></td>
          <td><code>void</code></td>
          <td>Synchronously or asynchronously copies device VRAM to host RAM.</td>
          <td>$O(MN)$</td>
        </tr>
        <tr>
          <td><code>float* data() noexcept</code></td>
          <td><code>float*</code></td>
          <td>Returns raw device pointer to underlying VRAM memory buffer.</td>
          <td>$O(1)$</td>
        </tr>
        <tr>
          <td><code>size_t rows() const noexcept</code></td>
          <td><code>size_t</code></td>
          <td>Returns row count $M$.</td>
          <td>$O(1)$</td>
        </tr>
        <tr>
          <td><code>size_t cols() const noexcept</code></td>
          <td><code>size_t</code></td>
          <td>Returns column count $N$.</td>
          <td>$O(1)$</td>
        </tr>
      </tbody>
    </table>

    <div class="page-subbreak"></div>

    <h2>13.2 Mathematical & Linear Algebra Operators</h2>
    <table>
      <thead>
        <tr>
          <th>Method Signature</th>
          <th>Return Type</th>
          <th>Description & Mathematical Rule</th>
          <th>Complexity</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><code>operator*(const Matrix& other) const</code></td>
          <td><code>Matrix</code></td>
          <td>GEMM matrix multiplication $\mathbf{C} = \mathbf{A} \times \mathbf{B}$ via adaptive dispatch.</td>
          <td>$O(MNK)$</td>
        </tr>
        <tr>
          <td><code>static void multiply_into(Matrix& C, const Matrix& A, const Matrix& B)</code></td>
          <td><code>void</code></td>
          <td>Allocation-free hot-path GEMM storing directly into preallocated $C$.</td>
          <td>$O(MNK)$</td>
        </tr>
        <tr>
          <td><code>operator+(const Matrix& other) const</code></td>
          <td><code>Matrix</code></td>
          <td>Element-wise addition $C_{ij} = A_{ij} + B_{ij}$ with automatic broadcasting.</td>
          <td>$O(MN)$</td>
        </tr>
        <tr>
          <td><code>operator-(const Matrix& other) const</code></td>
          <td><code>Matrix</code></td>
          <td>Element-wise subtraction $C_{ij} = A_{ij} - B_{ij}$.</td>
          <td>$O(MN)$</td>
        </tr>
        <tr>
          <td><code>hadamard(const Matrix& other) const</code></td>
          <td><code>Matrix</code></td>
          <td>Element-wise Schur product $C_{ij} = A_{ij} \cdot B_{ij}$.</td>
          <td>$O(MN)$</td>
        </tr>
        <tr>
          <td><code>transpose() const</code></td>
          <td><code>Matrix</code></td>
          <td>Matrix transpose $\mathbf{A}^T$ via conflict-free shared memory tiles.</td>
          <td>$O(MN)$</td>
        </tr>
        <tr>
          <td><code>relu() const</code></td>
          <td><code>Matrix</code></td>
          <td>Branchless rectified linear unit: $\max(0, x)$.</td>
          <td>$O(MN)$</td>
        </tr>
        <tr>
          <td><code>gelu() const</code></td>
          <td><code>Matrix</code></td>
          <td>Gaussian Error Linear Unit polynomial approximation.</td>
          <td>$O(MN)$</td>
        </tr>
        <tr>
          <td><code>sigmoid() const</code></td>
          <td><code>Matrix</code></td>
          <td>Logistic sigmoid: $\sigma(x) = \frac{1}{1 + e^{-x}}$.</td>
          <td>$O(MN)$</td>
        </tr>
        <tr>
          <td><code>softmax(int dim = -1) const</code></td>
          <td><code>Matrix</code></td>
          <td>Numerically stable three-pass online exponential normalization.</td>
          <td>$O(MN)$</td>
        </tr>
        <tr>
          <td><code>sum(int dim = -1) const</code></td>
          <td><code>Matrix</code></td>
          <td>Warp-shuffle tree reduction along dimension (or entire matrix if -1).</td>
          <td>$O(MN)$</td>
        </tr>
        <tr>
          <td><code>mean(int dim = -1) const</code></td>
          <td><code>Matrix</code></td>
          <td>Warp-shuffle tree mean reduction.</td>
          <td>$O(MN)$</td>
        </tr>
        <tr>
          <td><code>lu_factorize() const</code></td>
          <td><code>std::pair&lt;Matrix, vector&lt;int&gt;&gt;</code></td>
          <td>LU decomposition with partial pivoting $\mathbf{PA} = \mathbf{LU}$.</td>
          <td>$O(N^3)$</td>
        </tr>
        <tr>
          <td><code>cholesky() const</code></td>
          <td><code>Matrix</code></td>
          <td>Cholesky decomposition $\mathbf{A} = \mathbf{LL}^T$ for SPD matrices.</td>
          <td>$O(N^3)$</td>
        </tr>
        <tr>
          <td><code>qr() const</code></td>
          <td><code>std::pair&lt;Matrix, Matrix&gt;</code></td>
          <td>QR factorization $\mathbf{A} = \mathbf{QR}$ via Householder reflectors.</td>
          <td>$O(MN^2)$</td>
        </tr>
        <tr>
          <td><code>svd() const</code></td>
          <td><code>std::tuple&lt;Matrix, Matrix, Matrix&gt;</code></td>
          <td>Singular Value Decomposition $\mathbf{A} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$.</td>
          <td>$O(MN \min(M, N))$</td>
        </tr>
        <tr>
          <td><code>inverse() const</code></td>
          <td><code>Matrix</code></td>
          <td>Matrix inversion $\mathbf{A}^{-1}$ via LU factorization and TRSM solve.</td>
          <td>$O(N^3)$</td>
        </tr>
        <tr>
          <td><code>expm() const</code></td>
          <td><code>Matrix</code></td>
          <td>Matrix exponential $e^{\mathbf{A}}$ via Padé [13/13] scaling and squaring.</td>
          <td>$O(N^3)$</td>
        </tr>
      </tbody>
    </table>

    <div class="page-subbreak"></div>

    <h2>13.3 Autograd & Dynamic Differentiation Interface</h2>
    <table>
      <thead>
        <tr>
          <th>Class / Method Signature</th>
          <th>Return Type</th>
          <th>Description</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><code>Variable(Matrix data, bool requires_grad = true)</code></td>
          <td>Instance</td>
          <td>Wraps a Matrix into a dynamic computation graph node.</td>
        </tr>
        <tr>
          <td><code>void Variable::backward()</code></td>
          <td><code>void</code></td>
          <td>Initiates reverse-mode topological gradient propagation from this loss variable.</td>
        </tr>
        <tr>
          <td><code>const Matrix& Variable::grad() const</code></td>
          <td><code>const Matrix&</code></td>
          <td>Returns accumulated gradient tensor $\frac{\partial \mathcal{L}}{\partial \mathbf{x}}$.</td>
        </tr>
        <tr>
          <td><code>void Variable::zero_grad()</code></td>
          <td><code>void</code></td>
          <td>Resets accumulated gradients to zero via <code>cudaMemsetAsync</code>.</td>
        </tr>
      </tbody>
    </table>

    <h2>13.4 Sparse Matrix Interface: matrix_pro::sparse</h2>
    <table>
      <thead>
        <tr>
          <th>Class / Method Signature</th>
          <th>Return Type</th>
          <th>Description</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><code>CsrMatrix::from_dense(const Matrix& dense, float eps = 1e-6f)</code></td>
          <td><code>CsrMatrix</code></td>
          <td>Compacts dense matrix into CSR format on GPU via parallel stream compaction.</td>
        </tr>
        <tr>
          <td><code>CsrMatrix::from_coo(const CooMatrix& coo)</code></td>
          <td><code>CsrMatrix</code></td>
          <td>Converts COO coordinate triplets to CSR format via prefix scan.</td>
        </tr>
        <tr>
          <td><code>Matrix CsrMatrix::spmm(const Matrix& B) const</code></td>
          <td><code>Matrix</code></td>
          <td>Warp-per-row sparse-dense matrix multiplication $\mathbf{C} = \mathbf{A}_{\text{csr}} \times \mathbf{B}_{\text{dense}}$.</td>
        </tr>
        <tr>
          <td><code>CsrMatrix CsrMatrix::spgemm(const CsrMatrix& B) const</code></td>
          <td><code>CsrMatrix</code></td>
          <td>Gustavson two-phase sparse-sparse multiplication $\mathbf{C} = \mathbf{A} \times \mathbf{B}$.</td>
        </tr>
      </tbody>
    </table>

    <h2>13.5 CUDA Runtime & Execution Engine: matrix_pro::cuda</h2>
    <table>
      <thead>
        <tr>
          <th>Class / Method Signature</th>
          <th>Return Type</th>
          <th>Description</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><code>Stream(Stream::Flags flags = Flags::NonBlocking)</code></td>
          <td>Instance</td>
          <td>RAII wrapper managing independent CUDA execution stream.</td>
        </tr>
        <tr>
          <td><code>void Stream::synchronize()</code></td>
          <td><code>void</code></td>
          <td>Blocks host CPU until all queued device stream operations complete.</td>
        </tr>
        <tr>
          <td><code>void Event::record(const Stream& stream)</code></td>
          <td><code>void</code></td>
          <td>Places hardware timestamp marker into active execution queue.</td>
        </tr>
        <tr>
          <td><code>float Event::elapsed_time(const Event& start, const Event& end)</code></td>
          <td><code>float</code></td>
          <td>Computes exact GPU execution time in milliseconds (0.5 &mu;s precision).</td>
        </tr>
        <tr>
          <td><code>void CudaGraph::capture(Stream& stream, Func&& workload)</code></td>
          <td><code>void</code></td>
          <td>Captures multi-kernel sequence into an instantiated executable graph.</td>
        </tr>
        <tr>
          <td><code>void CudaGraph::replay(Stream& stream)</code></td>
          <td><code>void</code></td>
          <td>Instantly dispatches entire captured DAG to GPU in &lt; 2 &mu;s launch latency.</td>
        </tr>
        <tr>
          <td><code>MemoryPool::instance().allocate(size_t bytes, Stream& stream)</code></td>
          <td><code>void*</code></td>
          <td>Sub-allocates memory from pre-cached VRAM arena bins in &lt; 2 &mu;s.</td>
        </tr>
      </tbody>
    </table>
  </div>
`;

