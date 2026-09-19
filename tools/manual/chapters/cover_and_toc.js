// tools/manual/chapters/cover_and_toc.js

module.exports = `
  <!-- COVER PAGE -->
  <div class="cover-page">
    <div class="cover-header">
      <div class="cover-badge">Enterprise Technical Whitepaper & Specification</div>
      <h1 class="cover-title">MatrixFlash<span>-Pro</span></h1>
      <div class="cover-subtitle">
        High-Performance GPU-Accelerated C++17 Matrix & Tensor Computation Engine Architecture
      </div>
      <p style="color: #CBD5E1; font-size: 10pt; max-width: 650px; text-align: left;">
        Comprehensive architectural breakdown, mathematical formulations, hardware micro-optimizations, algorithmic foundations, complete API directory, and production machine learning benchmarks on modern NVIDIA architectures.
      </p>
    </div>

    <div class="cover-spec-grid">
      <div class="cover-spec-card">
        <div class="label">Target Architectures</div>
        <div class="val">sm_70 — sm_90</div>
        <div style="font-size: 7.5pt; color: #94A3B8; margin-top: 2px;">Volta, Turing, Ampere, Ada, Hopper</div>
      </div>
      <div class="cover-spec-card">
        <div class="label">Peak Measured GEMM</div>
        <div class="val">16,895 GFLOPS</div>
        <div style="font-size: 7.5pt; color: #94A3B8; margin-top: 2px;">RTX 3070 sm_86 (cuBLAS: 16,882)</div>
      </div>
      <div class="cover-spec-card">
        <div class="label">Memory Alloc Overhead</div>
        <div class="val">&lt; 0.002 ms</div>
        <div style="font-size: 7.5pt; color: #94A3B8; margin-top: 2px;">Lock-free arena block sub-allocator</div>
      </div>
      <div class="cover-spec-card">
        <div class="label">Language Standard</div>
        <div class="val">Modern C++17</div>
        <div style="font-size: 7.5pt; color: #94A3B8; margin-top: 2px;">Strict RAII, zero-cost abstractions</div>
      </div>
      <div class="cover-spec-card">
        <div class="label">Graph Dispatch Latency</div>
        <div class="val">&lt; 2.0 &mu;s</div>
        <div style="font-size: 7.5pt; color: #94A3B8; margin-top: 2px;">CUDA Graph instantiated replay</div>
      </div>
      <div class="cover-spec-card">
        <div class="label">Core Engine Modules</div>
        <div class="val">11 Subsystems</div>
        <div style="font-size: 7.5pt; color: #94A3B8; margin-top: 2px;">Tensor, GEMM, Autograd, Sparse, RNG...</div>
      </div>
    </div>

    <div class="cover-footer">
      <div>Author: MatrixFlash-Pro Engineering Team</div>
      <div>Document Version: 2.4.0-PRO</div>
      <div>Classification: Open Technical Standard</div>
      <div>Publication Date: September 2026</div>
    </div>
  </div>

  <!-- EXECUTIVE SUMMARY -->
  <div class="chapter">
    <div class="chapter-header">
      <div class="chapter-num">System Overview</div>
      <h1 class="chapter-title">Executive Summary & Architectural Manifesto</h1>
    </div>

    <h2>1. The High-Performance Computing Paradox</h2>
    <p>
      In modern deep learning and scientific computing, vendor-provided mathematical kernels (such as NVIDIA cuBLAS, cuSOLVER, and cuDNN) deliver peak theoretical arithmetic throughput. However, real-world application pipelines rarely achieve this ceiling. The performance degradation stems from critical structural overheads:
    </p>
    <ul>
      <li><strong>Allocation Stalls:</strong> Conventional frameworks perform dynamic device allocations (<code>cudaMalloc</code>) within hot training loops. Each call incurs an OS/driver context trap, serializing threads and forcing pipeline flushes.</li>
      <li><strong>Intermediate Roundtrips:</strong> Standard computation models evaluate operations sequentially. An expression such as <code>Y = ReLU(X * W + b)</code> writes large intermediate matrices back to high-latency global device memory (VRAM) multiple times, creating severe memory bus saturation.</li>
      <li><strong>CPU Launch Overhead:</strong> Issuing thousands of tiny CUDA kernels over PCIe causes the host CPU to become the bottleneck, leaving modern GPUs with 40-140 Streaming Multiprocessors (SMs) idle between kernel launches.</li>
    </ul>

    <h2>2. The MatrixFlash-Pro Mission</h2>
    <p>
      <strong>MatrixFlash-Pro</strong> is engineered from first principles to resolve these bottlenecks. Written in pure <strong>C++17</strong> with low-level CUDA assembly and runtime intrinsics, the library provides a unified, expressive mathematical tensor interface while enforcing zero-cost performance guarantees:
    </p>

    <div class="grid-2">
      <div class="callout tip">
        <div class="callout-title">✓ Adaptive Shape-Aware GEMM</div>
        Dynamically selects between register-tiled micro-kernels (small matrices), Tensor-Core accelerated cuBLAS GemmEx (medium), and cublasLt heuristic algorithm search with 32 MiB workspace caching (large matrices).
      </div>
      <div class="callout tip">
        <div class="callout-title">✓ Fused Epilogue Pipelines</div>
        Performs Matrix Multiply + Bias Addition + Activation (ReLU, GELU, SiLU) in a single kernel pass, completely eliminating intermediate global memory roundtrips and boosting throughput up to 6&times;.
      </div>
      <div class="callout tip">
        <div class="callout-title">✓ Zero-Allocation Hot Path</div>
        The <code>multiply_into()</code> primitive and custom block arena pool bypass device allocation locks, matching or outperforming raw vendor cuBLAS benchmarks across standard sizes.
      </div>
      <div class="callout tip">
        <div class="callout-title">✓ Sub-Microsecond Graph Dispatch</div>
        Full CUDA Graph capture and replay converts multi-operator autograd training graphs into single-node GPU executable plans, dropping dispatch overhead below 2 microseconds.
      </div>
    </div>

    <h2>3. The 11 Unified Engine Architecture</h2>
    <div class="arch-diagram">
+---------------------------------------------------------------------------------------------------+
|                                  USER APPLICATION & ALGORITHMS                                    |
|             (Deep Learning / Transformer Attention / Scientific Solvers / GNN Message Passing)    |
+---------------------------------------------------------------------------------------------------+
  |                  |                   |                  |                   |                 |
  v                  v                   v                  v                   v                 v
+-------------+  +---------------+  +---------------+  +---------------+  +---------------+  +------------+
| 1. TENSOR   |  | 2. MATH & OP  |  | 3. LIN. ALG.  |  | 4. AUTOGRAD   |  | 5. SPARSE     |  | 6. RNG     |
| Strided N-D |  | FMA, Activat. |  | GEMM, SVD, LU |  | Dynamic Tape  |  | CSR, COO, CSC |  | Philox4x32 |
| Zero-Copy   |  | Warp-Shuffle  |  | Padé Expm     |  | Checkpointing |  | SpMM, SpGEMM  |  | Box-Muller |
+-------------+  +---------------+  +---------------+  +---------------+  +---------------+  +------------+
  |                  |                   |                  |                   |                 |
  +------------------+-------------------+------------------+-------------------+-----------------+
                                         |
                                         v
+---------------------------------------------------------------------------------------------------+
| 7. MEMORY ENGINE      | Fast Arena Block Pool | Unified Memory Prefetch | Pinned Async Transfers  |
+---------------------------------------------------------------------------------------------------+
| 8. EXECUTION ENGINE   | Multi-Stream Overlap  | Event Profiling         | CUDA Graph Replay Engine|
+---------------------------------------------------------------------------------------------------+
| 9. BACKEND ENGINE     | Heterogeneous Routing | CPU OpenBLAS Fallback   | Multi-GPU P2P NVLink    |
+---------------------------------------------------------------------------------------------------+
| 10. SERIALIZATION     | Native .mflash Binary | Zero-Copy NumPy .npy    | SafeTensors Checkpoint  |
+---------------------------------------------------------------------------------------------------+
| 11. BENCHMARK ENGINE  | Hardware CUDA Events  | Roofline Profiling      | Flop Counter & Telemetry|
+---------------------------------------------------------------------------------------------------+
    </div>
  </div>

  <!-- TABLE OF CONTENTS -->
  <div class="chapter">
    <div class="chapter-header">
      <div class="chapter-num">Index</div>
      <h1 class="chapter-title">Table of Contents</h1>
    </div>

    <div class="toc-entry"><span class="title">1. Hardware Foundations & NVIDIA GPU Execution Model</span><span class="dots"></span><span class="num">04</span></div>
    <div class="toc-entry"><span class="title">2. Tensor Engine: Multi-Dimensional Strided Math</span><span class="dots"></span><span class="num">08</span></div>
    <div class="toc-entry"><span class="title">3. Mathematical & Element-Wise Operator Engine</span><span class="dots"></span><span class="num">12</span></div>
    <div class="toc-entry"><span class="title">4. High-Performance Linear Algebra & GEMM Engine</span><span class="dots"></span><span class="num">16</span></div>
    <div class="toc-entry"><span class="title">5. Advanced Decompositions & Direct Solvers (LU, SVD, QR, Expm)</span><span class="dots"></span><span class="num">20</span></div>
    <div class="toc-entry"><span class="title">6. Memory Engine: Sub-Allocators & Unified Memory Architecture</span><span class="dots"></span><span class="num">24</span></div>
    <div class="toc-entry"><span class="title">7. Backend Engine: Heterogeneous Compute & Multi-GPU P2P</span><span class="dots"></span><span class="num">28</span></div>
    <div class="toc-entry"><span class="title">8. Execution Engine: Asynchronous Streams & CUDA Graph Replay</span><span class="dots"></span><span class="num">31</span></div>
    <div class="toc-entry"><span class="title">9. Sparse Matrix Engine: Storage Schemes & SpMM / SpGEMM Kernels</span><span class="dots"></span><span class="num">35</span></div>
    <div class="toc-entry"><span class="title">10. Automatic Differentiation Engine (Dynamic Tape Autograd)</span><span class="dots"></span><span class="num">39</span></div>
    <div class="toc-entry"><span class="title">11. Pseudo-Random Number Engine: Counter-Based Stateless PRNG</span><span class="dots"></span><span class="num">43</span></div>
    <div class="toc-entry"><span class="title">12. Serialization Engine: Native .mflash & Zero-Copy SafeTensors</span><span class="dots"></span><span class="num">46</span></div>
    <div class="toc-entry"><span class="title">13. Complete API Reference Directory</span><span class="dots"></span><span class="num">49</span></div>
    <div class="toc-entry"><span class="title">14. Enterprise Production Examples & Verification Implementations</span><span class="dots"></span><span class="num">52</span></div>
    <div class="toc-entry"><span class="title">15. Empirical Benchmark Verification & Competitive Analysis</span><span class="dots"></span><span class="num">55</span></div>
    <div class="toc-entry"><span class="title">16. Low-Level PTX Assembly & SASS Instruction Scheduling</span><span class="dots"></span><span class="num">57</span></div>
    <div class="toc-entry"><span class="title">17. Formal Mathematical Proofs & Numerical Stability</span><span class="dots"></span><span class="num">59</span></div>
    <div class="toc-entry"><span class="title">18. Enterprise Production Deployment, C FFI & Docker</span><span class="dots"></span><span class="num">61</span></div>

    <div class="callout warning" style="margin-top: 24px;">
      <div class="callout-title">Document Scope and Target Audience</div>
      This specification is intended for high-performance computing engineers, AI framework developers, quantitative researchers, and CUDA systems architects. It details both theoretical mathematical derivations and physical register-level machine code execution patterns.
    </div>
  </div>
`;
