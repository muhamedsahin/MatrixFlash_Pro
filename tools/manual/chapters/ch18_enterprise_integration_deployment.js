// tools/manual/chapters/ch18_enterprise_integration_deployment.js

module.exports = `
  <div class="chapter">
    <div class="chapter-header">
      <div class="chapter-num">Chapter 18</div>
      <h1 class="chapter-title">Enterprise Production Deployment, C FFI & Docker</h1>
    </div>

    <h2>18.1 Modern CMake Integration & Build System</h2>
    <p>
      MatrixFlash-Pro is designed for frictionless integration into modern C++ enterprise applications via standard CMake target exporting. In your downstream <code>CMakeLists.txt</code>:
    </p>

    <pre><code>cmake_minimum_required(VERSION 3.22)
project(EnterpriseInferenceService LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")

# Locate MatrixFlash-Pro package installation
find_package(MatrixFlashPro CONFIG REQUIRED)

add_executable(inference_server main.cpp)
target_link_libraries(inference_server PRIVATE matrix_pro::matrix_pro)</code></pre>

    <h2>18.2 C Foreign Function Interface (FFI) for Polyglot Bindings</h2>
    <p>
      To enable zero-overhead interoperability with Python, Rust, Go, C#, and Julia, MatrixFlash-Pro exposes a pure C ABI (Application Binary Interface) free of C++ name mangling:
    </p>

    <pre><code>// matrix_flash_c_api.h - Pure C Foreign Function Interface
#ifndef MATRIX_FLASH_C_API_H
#define MATRIX_FLASH_C_API_H

#include &lt;stddef.h&gt;
#include &lt;stdint.h&gt;

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MFlashMatrix* MFlashMatrixHandle;

int mflash_matrix_create(size_t rows, size_t cols, MFlashMatrixHandle* out_handle);
int mflash_matrix_destroy(MFlashMatrixHandle handle);
int mflash_matrix_matmul(MFlashMatrixHandle A, MFlashMatrixHandle B, MFlashMatrixHandle* out_C);
int mflash_matrix_multiply_into(MFlashMatrixHandle C, MFlashMatrixHandle A, MFlashMatrixHandle B);
int mflash_matrix_download(MFlashMatrixHandle handle, float* host_buffer);

#ifdef __cplusplus
}
#endif
#endif</code></pre>

    <div class="page-subbreak"></div>

    <h2>18.3 Production Docker Containerization with NVIDIA Container Toolkit</h2>
    <p>
      For enterprise Kubernetes and cloud deployment (AWS EC2, Google Cloud GKE, Azure AKS), MatrixFlash-Pro provides an optimized multi-stage <code>Dockerfile</code>:
    </p>

    <pre><code># Multi-Stage Production GPU Container
FROM nvidia/cuda:12.3.2-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential cmake ninja-build git && \\
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Compile optimized release binaries with multi-architecture PTX
RUN cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \\
    -DCMAKE_CUDA_ARCHITECTURES="70;80;86;90" && \\
    cmake --build build --target matrix_pro_basic matrix_pro_bench_comparison

# Lightweight Runtime Image
FROM nvidia/cuda:12.3.2-runtime-ubuntu22.04

WORKDIR /workspace
COPY --from=builder /app/build/bin/* /usr/local/bin/

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

ENTRYPOINT ["/usr/local/bin/matrix_pro_basic"]</code></pre>

    <div class="callout tip">
      <div class="callout-title">Enterprise Deployment Command</div>
      To launch with full GPU acceleration on enterprise clusters:<br>
      <code>docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it matrixflash-pro:latest</code>
    </div>
  </div>
`;

