// tools/manual/chapters/ch07_backend_and_heterogeneous.js

module.exports = `
  <div class="chapter">
    <div class="chapter-header">
      <div class="chapter-num">Chapter 07</div>
      <h1 class="chapter-title">Backend Engine: Heterogeneous Compute & Multi-GPU P2P</h1>
    </div>

    <h2>7.1 Heterogeneous Device Abstraction Layer</h2>
    <p>
      High-performance enterprise applications require execution portability across both GPU-accelerated and CPU-only environments. The MatrixFlash-Pro <strong>Backend Engine</strong> decouples mathematical algorithms from underlying hardware architectures through an abstract <code>DeviceContext</code> interface:
    </p>

    <div class="arch-diagram">
                       +-----------------------------------+
                       |      matrix_pro::Matrix Class     |
                       +-----------------------------------+
                                         |
                                         v
                       +-----------------------------------+
                       |       Backend Engine Router       |
                       +-----------------------------------+
                        /                                 \
                       v                                   v
        +-----------------------------+     +-----------------------------+
        |        CUDA BACKEND         |     |         CPU BACKEND         |
        | - Custom PTX CUDA Kernels   |     | - OpenMP Multi-Threading    |
        | - cuBLAS, cuSOLVER, cuSPARSE|     | - AVX-512 / AVX2 SIMD       |
        | - NVLink P2P Direct DMA     |     | - OpenBLAS / MKL Fallback   |
        +-----------------------------+     +-----------------------------+
    </div>

    <pre><code>// Dynamic Backend Migration
Matrix a = Matrix::random(1024, 1024, DeviceType::CUDA); // Resides in VRAM
Matrix b = a.to(DeviceType::CPU);                         // Transparently staged to Host RAM
Matrix c = b.relu();                                      // Evaluated via AVX2 vector instructions
Matrix d = c.to(DeviceType::CUDA);                         // Pushed back to VRAM via DMA pipeline</code></pre>

    <h2>7.2 Multi-GPU Peer-to-Peer (P2P) Direct Memory Access</h2>
    <p>
      In multi-GPU workstations and enterprise DGX compute nodes, GPUs are interconnected via PCIe switches or high-speed <strong>NVIDIA NVLink</strong> bridges (up to 900 GB/s per GPU). MatrixFlash-Pro automatically queries the device topology and enables bidirectional <strong>Peer-to-Peer (P2P) Memory Access</strong>:
    </p>

    <div class="formula-box">
      $$\text{cudaDeviceCanAccessPeer}(\&amp;\text{canAccess}, \text{dev0}, \text{dev1}) \implies \text{cudaDeviceEnablePeerAccess}(\text{dev1}, 0)$$
      <span class="eq-desc">Allows GPU 0 to read/write GPU 1's physical memory over NVLink with zero CPU host involvement</span>
    </div>

    <pre><code>// Automated Multi-GPU Topology Discovery
void initialize_p2p_topology() {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    
    for (int i = 0; i &lt; device_count; ++i) {
        CUDA_CHECK(cudaSetDevice(i));
        for (int j = 0; j &lt; device_count; ++j) {
            if (i == j) continue;
            int can_access = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, i, j));
            if (can_access) {
                cudaError_t err = cudaDeviceEnablePeerAccess(j, 0);
                if (err == cudaSuccess || err == cudaErrorPeerAccessAlreadyEnabled) {
                    cudaGetLastError(); // Clear benign already-enabled status
                }
            }
        }
    }
}</code></pre>

    <div class="page-subbreak"></div>

    <h2>7.3 Distributed Collective Primitives: Ring AllReduce</h2>
    <p>
      For distributed deep learning across $P$ GPUs, model gradients must be summed and synchronized across all devices. Naïve parameter-server approaches cause network bottlenecking at the root node. MatrixFlash-Pro implements the mathematically optimal <strong>Ring AllReduce Algorithm</strong>:
    </p>

    <div class="arch-diagram">
+---------------------------------------------------------------------------------------------------+
|                            THE RING ALLREDUCE PIPELINE (P = 4 GPUs)                               |
+---------------------------------------------------------------------------------------------------+

     GPU 0 -----------------NVLink-----------------> GPU 1
       ^                                               |
       |                                               |
     NVLink                                          NVLink
       |                                               |
       v                                               v
     GPU 3 <----------------NVLink------------------ GPU 2

  Phase 1: Scatter-Reduce (P - 1 steps)
  Each GPU transmits 1/P slice of its tensor to its downstream neighbor, accumulating incoming data.
  Total Data Transmitted: 2 * ((P - 1) / P) * TensorSize (Independent of GPU count!)
    </div>

    <div class="formula-box">
      $$\text{Total Transferred Volume per GPU} = 2 \left( \frac{P - 1}{P} \right) \times \text{Size}_{\text{bytes}}$$
      <span class="eq-desc">Ring AllReduce communication volume is strictly constant with respect to cluster size</span>
    </div>

    <p>
      By pipelining chunk transmissions across independent CUDA streams over NVLink, the Ring AllReduce implementation saturates available hardware bandwidth, delivering linear scaling efficiency (>94%) up to 8 GPUs.
    </p>
  </div>
`;

