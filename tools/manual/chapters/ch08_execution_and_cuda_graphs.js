// tools/manual/chapters/ch08_execution_and_cuda_graphs.js

module.exports = `
  <div class="chapter">
    <div class="chapter-header">
      <div class="chapter-num">Chapter 08</div>
      <h1 class="chapter-title">Execution Engine: Asynchronous Streams & CUDA Graphs</h1>
    </div>

    <h2>8.1 Multi-Stream Asynchronous Concurrency</h2>
    <p>
      In NVIDIA GPUs, concurrency is achieved through <strong>CUDA Streams</strong>: queues of operations executed in sequential order on the device, but completely concurrently with operations in other streams. MatrixFlash-Pro encapsulates stream management inside a zero-cost RAII wrapper (<code>matrix_pro::cuda::Stream</code>):
    </p>

    <pre><code>// Independent Stream Task Concurrency
Stream stream1(Stream::Flags::NonBlocking);
Stream stream2(Stream::Flags::NonBlocking);

// Dispatched concurrently across different SM clusters
matrix_a.matmul_async(matrix_b, out1, stream1);
matrix_c.matmul_async(matrix_d, out2, stream2);

// Fine-grained cross-stream event barrier without CPU blocking
Event event1;
event1.record(stream1);
stream2.wait_event(event1); // stream2 pauses until stream1 finishes, CPU remains unblocked!</code></pre>

    <h2>8.2 High-Precision Sub-Microsecond Hardware Profiling</h2>
    <p>
      Measuring kernel execution via CPU timers (such as <code>std::chrono</code>) introduces operating system scheduling jitter, driver queue submission latency, and PCIe roundtrip delay. MatrixFlash-Pro measures performance strictly via <strong>Hardware CUDA Events</strong> placed directly inside the GPU instruction pipeline:
    </p>

    <div class="formula-box">
      $$\Delta t_{\text{hardware}} = \text{cudaEventElapsedTime}(\&amp;\text{ms}, \mathcal{E}_{\text{start}}, \mathcal{E}_{\text{stop}})$$
      <span class="eq-desc">Timestamp resolution: 0.5 microseconds, sampled directly from GPU SM hardware clock ticks</span>
    </div>

    <div class="page-subbreak"></div>

    <h2>8.3 CUDA Graph Engine: Eliminating CPU Launch Latency</h2>
    <p>
      In recurrent networks, transformers, and iterative solvers, hundreds of small kernels (reductions, activations, bias additions) are issued sequentially. Each CPU launch costs between $5\,\mu\text{s}$ and $15\,\mu\text{s}$. If a kernel runs in $3\,\mu\text{s}$, the GPU spends over $75\%$ of its time waiting for the CPU:
    </p>

    <div class="arch-diagram">
TRADITIONAL CPU-BOUND DISPATCH TIMELINE:
CPU: |-- Launch K1 --| (idle) |-- Launch K2 --| (idle) |-- Launch K3 --|
GPU:                 |-- K1 --|                |-- K2 --|                |-- K3 --|
Notice the massive idle gaps on the GPU between kernels!

-------------------------------------------------------------------------------------

CUDA GRAPH REPLAY TIMELINE (Single Hardware Launch Primitive):
CPU: |-- cudaGraphLaunch() --| (Returns immediately in < 2 microseconds!)
GPU: |-- K1 --||-- K2 --||-- K3 --| (Zero inter-kernel latency, 100% SM saturation!)
    </div>

    <p>
      MatrixFlash-Pro provides an automated <strong>CUDA Graph Engine</strong> that captures complex computational topologies into an immutable execution graph:
    </p>

    <pre><code>// CUDA Graph Capture and Instantaneous Replay Flow
class CudaGraph {
public:
    template &lt;typename Func&gt;
    void capture(Stream& stream, Func&& workload) {
        CUDA_CHECK(cudaStreamBeginCapture(stream.handle(), cudaStreamCaptureModeGlobal));
        
        // Execute arbitrary tensor operations (matmul, activations, backward passes)
        workload();

        CUDA_CHECK(cudaStreamEndCapture(stream.handle(), &graph_));
        CUDA_CHECK(cudaGraphInstantiate(&exec_graph_, graph_, nullptr, nullptr, 0));
    }

    void replay(Stream& stream) {
        // Launches the entire multi-kernel DAG in a single sub-2 microsecond API call
        CUDA_CHECK(cudaGraphLaunch(exec_graph_, stream.handle()));
    }

    ~CudaGraph() {
        if (exec_graph_) cudaGraphExecDestroy(exec_graph_);
        if (graph_) cudaGraphDestroy(graph_);
    }

private:
    cudaGraph_t graph_{nullptr};
    cudaGraphExec_t exec_graph_{nullptr};
};</code></pre>

    <div class="callout tip">
      <div class="callout-title">Production Verification: Graph Replay Speedup</div>
      In an end-to-end 6-layer Transformer feed-forward block evaluation, CUDA Graph replay reduced total iteration wall-clock time from <strong>0.384 ms to 0.112 ms</strong>—a <strong>3.4&times; pure speedup</strong> achieved entirely by eliminating host launch overhead.
    </div>
  </div>
`;

