// tools/manual/chapters/ch12_serialization_engine.js

module.exports = `
  <div class="chapter">
    <div class="chapter-header">
      <div class="chapter-num">Chapter 12</div>
      <h1 class="chapter-title">Serialization Engine: Native .mflash & Zero-Copy SafeTensors</h1>
    </div>

    <h2>12.1 The Native .mflash Binary File Specification</h2>
    <p>
      For enterprise model deployment and high-throughput checkpoint saving, serialization formats must support zero-overhead direct memory access (DMA). Standard formats (such as JSON, Pickle, or Protocol Buffers) impose parsing bottlenecks, non-contiguous allocations, and endianness ambiguities.
    </p>
    <p>
      MatrixFlash-Pro defines the <strong>.mflash</strong> binary specification, featuring a strict 64-byte aligned header followed immediately by raw binary tensor payloads:
    </p>

    <div class="arch-diagram">
+---------------------------------------------------------------------------------------------------+
|                           .MFLASH BINARY FILE CONTAINER STRUCTURE                                 |
+---------------------------------------------------------------------------------------------------+
| Offset (Bytes) | Field Name            | Data Type     | Description                              |
+----------------+-----------------------+---------------+------------------------------------------+
| 0x00 - 0x07    | Magic Signature       | char[8]       | ASCII: "MFLASH01"                        |
| 0x08 - 0x0B    | Endianness Tag        | uint32_t      | 0x01020304 (Validates Byte Order)        |
| 0x0C - 0x0F    | Data Type Code        | uint32_t      | 1=FP32, 2=FP16, 3=BF16, 4=INT32          |
| 0x10 - 0x13    | Tensor Rank (Dims)    | uint32_t      | Number of Dimensions D (1 to 8)          |
| 0x14 - 0x17    | Flags & Compression   | uint32_t      | Bit 0: Contiguous, Bit 1: Sparse CSR     |
| 0x18 - 0x1F    | Payload Byte Length   | uint64_t      | Exact total bytes of tensor data         |
| 0x20 - 0x23    | CRC-32 Checksum       | uint32_t      | IEEE 802.3 CRC over payload data         |
| 0x24 - 0x3F    | Reserved Padding      | uint8_t[28]   | Zero-filled (Ensures 64-byte alignment)  |
| 0x40 - ...     | Shape Tuple           | uint64_t[D]   | Dimension sizes S_0, S_1, ... S_{D-1}    |
| ...            | Stride Tuple          | int64_t[D]    | Stride values s_0, s_1, ... s_{D-1}      |
| Aligned 64B    | RAW PAYLOAD BUFFER    | float[...]    | Direct VRAM DMA Source / Destination     |
+---------------------------------------------------------------------------------------------------+
    </div>

    <pre><code>// Direct DMA File Deserialization
Matrix Matrix::load_mflash(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) throw std::runtime_error("Failed to open file: " + filepath);

    MFlashHeader header;
    file.read(reinterpret_cast&lt;char*&gt;(&header), sizeof(MFlashHeader));
    if (std::string(header.magic, 8) != "MFLASH01") {
        throw std::runtime_error("Invalid .mflash magic signature");
    }

    std::vector&lt;size_t&gt; shape(header.rank);
    file.read(reinterpret_cast&lt;char*&gt;(shape.data()), header.rank * sizeof(size_t));

    // Allocate directly on device
    Matrix result(shape[0], shape[1]);
    
    // Stage through pinned host buffer for async DMA copy
    PinnedHostBuffer host_buf(header.payload_bytes);
    file.read(reinterpret_cast&lt;char*&gt;(host_buf.data()), header.payload_bytes);

    CUDA_CHECK(cudaMemcpy(result.data(), host_buf.data(), header.payload_bytes, cudaMemcpyHostToDevice));
    return result;
}</code></pre>

    <div class="page-subbreak"></div>

    <h2>12.2 NumPy (.npy / .npz) Zero-Copy Interoperability</h2>
    <p>
      Python remains the lingua franca of machine learning experimentation. MatrixFlash-Pro provides native readers and writers for the NumPy <code>.npy</code> format without requiring Python, Cython, or external runtime libraries. The engine parses the Python dictionary header string:
    </p>

    <pre><code>// Example NumPy Header Parsed Internally
{'descr': '&lt;f4', 'fortran_order': False, 'shape': (2048, 2048), }</code></pre>

    <p>
      Once header dimensions and Fortran flags are extracted, the engine maps the binary block directly into GPU device memory via asynchronous DMA channels, achieving ingestion speeds of <strong>up to 14.8 GB/s over PCIe 4.0</strong>.
    </p>

    <h2>12.3 Hugging Face SafeTensors Checkpoint Loading</h2>
    <p>
      Modern open-weights foundation models (such as Llama-3, Mistral, and Stable Diffusion) are distributed in the <strong>SafeTensors</strong> format. SafeTensors files prepend a lightweight JSON header containing tensor names, shapes, and 64-bit byte offsets into an uncompressed binary payload:
    </p>

    <div class="formula-box">
      $$\text{Tensor Offset} = 8 + \text{HeaderLength} + \text{data\_offsets}[0]$$
      <span class="eq-desc">Enables direct memory-mapped file loading (<code>mmap</code>) into unified memory in zero CPU cycles</span>
    </div>

    <pre><code>// Loading Transformer Weights from SafeTensors Checkpoint
SafeTensorsArchive weights("llama-3-8b.safetensors");

// Zero-copy lookup and direct VRAM upload
Matrix q_proj = weights.load_tensor("model.layers.0.self_attn.q_proj.weight");
Matrix k_proj = weights.load_tensor("model.layers.0.self_attn.k_proj.weight");
Matrix v_proj = weights.load_tensor("model.layers.0.self_attn.v_proj.weight");

// Immediately ready for Tensor Core GEMM inference
Matrix q_out = x.matmul(q_proj);</code></pre>

    <div class="callout tip">
      <div class="callout-title">Security & Production Safety</div>
      Unlike Python <code>pickle</code> or PyTorch <code>.pt</code> files which allow arbitrary remote code execution via object deserialization, both <code>.mflash</code> and <code>SafeTensors</code> formats contain strictly typed numerical bytes and cannot execute malicious code.
    </div>
  </div>
`;

