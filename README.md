# MatrixFlash-Pro

C++17 matrix library. Host GEMM is `gemm_f32`. On this machine that call selects the AVX2 kernel `avx2-4x16`. Device GEMM is cuBLAS: `matrix_pro::hw::cuda_gemm` issues `cublasSgemm`, and the `GemmPlan` path uses cuBLAS or cuBLASLt.

License: Apache-2.0. The license text is the `LICENSE` file in this directory.

Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.

## Install

Tested host tools for this tree: CMake 3.24 or newer, and Visual Studio 2022 (MSVC). CUDA is optional. The measurements below used CUDA 13.4 when the GPU path was built. CPU GEMM, the thread pool, and the heterogeneous scheduler still build with CUDA turned off.

Preset build (CUDA on, if the toolkit is on `PATH`):

```powershell
cmake --preset release
cmake --build --preset release
ctest --preset release-tests
```

The local performance script finds Visual Studio’s `vcvars64.bat`, configures Ninja, and force-disables cuDNN:

```powershell
python tools/build_local.py --reconfigure --jobs 4
```

That script passes `-DMATRIX_PRO_ENABLE_CUDNN=OFF`. Convolution in that build is the CUDA kernel in `src/nn/conv.cu`, not cuDNN.

CPU-only configure:

```powershell
cmake -S . -B build-cpu -DMATRIX_PRO_ENABLE_CUDA=OFF
cmake --build build-cpu --config Release
```

Link the `matrix_pro` target. Public CPU headers are under `include/`.

## `gemm_f32`

Declared in `include/matrix_pro/cpu/gemm.hpp`:

```cpp
void gemm_f32(int m, int n, int k, const float* a, int lda, const float* b, int ldb,
              float* c, int ldc, float alpha, float beta);
```

Row-major `C = alpha * A * B + beta * C`. `A` is `m×k` with stride `lda`, `B` is `k×n` with stride `ldb`, `C` is `m×n` with stride `ldc`.

```cpp
#include "matrix_pro/cpu/gemm.hpp"

#include <vector>

int main() {
    const int m = 2;
    const int n = 2;
    const int k = 2;
    const std::vector<float> a{1.f, 2.f, 3.f, 4.f};
    const std::vector<float> b{5.f, 6.f, 7.f, 8.f};
    std::vector<float> c(static_cast<std::size_t>(m * n), 0.f);
    matrix_pro::gemm_f32(m, n, k, a.data(), k, b.data(), n, c.data(), n, 1.f, 0.f);
}
```

`gemm_kernel_name()` returns the selected kernel. On this CPU it is `avx2-4x16`.

## Device GEMM is cuBLAS

`include/matrix_pro/detail/cuda_hooks.hpp`:

```cpp
namespace matrix_pro::hw {
void cuda_gemm(int device, int m, int n, int k, const float* a, int lda, const float* b, int ldb,
               float* c, int ldc, float alpha, float beta);
}
```

The pointers are device pointers. The body in `src/hw/cuda_backend.cu` calls `cublasSgemm`. Without the CUDA toolkit the same signature throws `MatrixProError`.

`GemmPlan` / `gemm` / `gemm_into` in `include/matrix_pro/ops/gemm.hpp` are the Matrix-level device API. Their large path is cuBLAS or cuBLASLt. The default `GemmPrecision` is strict FP32. TF32 is an explicit option, not a separate library.

## This machine, evening of 2026-10-21

CPU: AMD Ryzen 7 5800H. It has no AVX-512, and AVX-512 is not selected. The serial AVX-512 kernel does not distribute rows, so `gemm_f32` keeps AVX2 when AVX2 and FMA are present.

Square GEMM, `N=2048`, this protocol:

| Run | GFLOPS |
|---|---:|
| Nexus | 330 |
| Nexus | 362 |
| Nexus | 369 |
| TensorFlow | 389 |

These three Nexus rates are the samples from that evening. They are not a world ranking. No other GFLOPS figure is implied.

Kept CPU kernel: `avx2-4x16` (B panel packed once).

Deleted, slower, and not the current kernel:

- an `Nc` tile that repacks A
- the MASM 6×16 kernel
- hybrid GEMM at `N=512`

cuDNN is force-disabled in the measured build.

CUDA conv versus this library’s CPU conv, same machine. Shape `8×3×32×32`, 16 filters, `3×3`, padding 1:

| Path | GFLOPS |
|---|---:|
| Nexus CPU conv | 6.39 |
| Device-resident CUDA kernel | 48.7 |
| Host round-trip | 19.0 |

That CUDA figure is not a comparison with PyTorch.

The 1 October 2026 GPU tables in `PERFORMANCE_REPORT.md` are an older cuBLAS protocol on an RTX 3070 Laptop. They are not this CPU kernel, and they are not a ranking of libraries.

## Manual

`doc/manual.md` is the technical manual source (install, GEMM, cuBLAS, the heterogeneous scheduler, the conv2d CUDA kernel, and the deleted experiments). Pandoc is not installed. The existing PDF files were not regenerated:

- `MatrixFlash_Pro_Technical_Manual.pdf`
- `doc/public/MatrixFlash_Pro_Technical_Manual.pdf`
