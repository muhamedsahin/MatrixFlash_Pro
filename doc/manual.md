# MatrixFlash-Pro technical manual

This file is the manual source. Pandoc is not installed, so the existing PDF binaries were left unchanged:

- `../MatrixFlash_Pro_Technical_Manual.pdf`
- `public/MatrixFlash_Pro_Technical_Manual.pdf`

License: Apache-2.0 (`../LICENSE`). This manual does not restate the license terms.

Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.

## 1. Install

Requirements used for this tree:

- CMake 3.24 or newer (`cmake_minimum_required` in the root `CMakeLists.txt`)
- Visual Studio 2022, so `cl.exe` is available
- CUDA 13.4 when the GPU translation units are built

CUDA is optional. `MATRIX_PRO_ENABLE_CUDA` defaults to `ON`. `OFF` still builds CPU GEMM, the thread pool, collectives, and the heterogeneous scheduler.

```powershell
cmake --preset release
cmake --build --preset release
ctest --preset release-tests
```

`tools/build_local.py` locates Visual Studio’s `vcvars64.bat`, configures Ninja, and passes `-DMATRIX_PRO_ENABLE_CUDNN=OFF`. That is the force-disabled cuDNN configuration. The measured conv2d path is the CUDA kernel, not cuDNN.

```powershell
python tools/build_local.py --reconfigure --jobs 4
```

CPU-only:

```powershell
cmake -S . -B build-cpu -DMATRIX_PRO_ENABLE_CUDA=OFF
cmake --build build-cpu --config Release
```

Consumers link `matrix_pro` and add `include/` to the header path.

## 2. Host GEMM

Header: `include/matrix_pro/cpu/gemm.hpp`. Also pulled in by `include/matrix_pro/hardware.hpp`.

```cpp
void gemm_f32(int m, int n, int k, const float* a, int lda, const float* b, int ldb,
              float* c, int ldc, float alpha, float beta);
```

`C = alpha * A * B + beta * C`, row-major. `lda` is the stride of `A` (`m×k`), `ldb` of `B` (`k×n`), `ldc` of `C` (`m×n`).

```cpp
#include "matrix_pro/cpu/gemm.hpp"

#include <vector>

int main() {
    const int n = 2048;
    std::vector<float> a(static_cast<std::size_t>(n) * n, 1.f);
    std::vector<float> b(static_cast<std::size_t>(n) * n, 1.f);
    std::vector<float> c(static_cast<std::size_t>(n) * n, 0.f);
    matrix_pro::gemm_f32(n, n, n, a.data(), n, b.data(), n, c.data(), n, 1.f, 0.f);
}
```

Selection lives in `src/hw/gemm.cpp`. If AVX2 and FMA are present, the call uses `gemm_avx2_serial`. `gemm_kernel_name()` then returns `avx2-4x16`. The 4×16 microkernel packs the B panel once. A scalar `gemm_reference` remains for machines without AVX2, and for `k <= 0`.

### AVX-512 is not selected on this CPU

The machine is an AMD Ryzen 7 5800H. It does not implement AVX-512. Independently, the serial AVX-512 kernel is not chosen while AVX2 is available: `kAvx512DistributesRows` is false, because that kernel does not distribute rows and it repacks A on every N tile. A CPUID flag alone does not switch the GEMM. `device_name()` may return the string `CPU AVX-512` only when `avx512f` is set; that string is not the GEMM selector, and it is not the name on this CPU.

### Evening of 2026-10-21, N=2048

Same machine, no AVX-512. Three Nexus samples and one TensorFlow sample:

| Implementation | GFLOPS |
|---|---:|
| Nexus | 330 |
| Nexus | 362 |
| Nexus | 369 |
| TensorFlow | 389 |

No world ranking is drawn from one laptop. No further GFLOPS values are stated for this protocol.

## 3. Device GEMM is cuBLAS

Header: `include/matrix_pro/detail/cuda_hooks.hpp`.

```cpp
namespace matrix_pro::hw {
void cuda_gemm(int device, int m, int n, int k, const float* a, int lda,
               const float* b, int ldb, float* c, int ldc, float alpha, float beta);
}
```

`src/hw/cuda_backend.cu` implements it with `cublasSgemm` (row-major `C = A*B` issued as column-major `C^T = B^T * A^T`). `a`, `b`, and `c` are device pointers. The call sets the device, locks that device’s slot, and uses the slot’s cuBLAS handle.

```cpp
#include "matrix_pro/detail/cuda_hooks.hpp"

void device_product(int device, int m, int n, int k,
                    const float* a_device, int lda,
                    const float* b_device, int ldb,
                    float* c_device, int ldc) {
    matrix_pro::hw::cuda_gemm(device, m, n, k, a_device, lda, b_device, ldb,
                              c_device, ldc, 1.f, 0.f);
}
```

The Matrix API is separate and also vendor GEMM. `include/matrix_pro/ops/gemm.hpp`:

```cpp
void gemm_into(const Matrix& left, const Matrix& right, Matrix& output,
               GemmOptions options = {}, float alpha = 1, float beta = 0,
               const Matrix* bias = nullptr);
```

`GemmPlan::execute` has the same buffer contract. Default precision is `GemmPrecision::fp32`. `GemmPrecision::tf32` is opt-in. The shape dispatcher in `include/matrix_pro/detail/gemm/gemm_api.hpp` can call cuBLAS (`gemm_cublas_ex`) or cuBLASLt (`gemm_cublas_lt`). Small shapes may use the library’s micro, GEMV, or tiled kernels; that does not replace cuBLAS on the `hw::cuda_gemm` entry.

## 4. Heterogeneous scheduler

Header: `include/matrix_pro/core/heterogeneous_scheduler.hpp`.

```cpp
void gemm(const float* a, int lda, const float* b, int ldb, float* c, int ldc,
          int m, int n, int k, float alpha = 1.0f, float beta = 0.0f);
```

Argument order differs from `gemm_f32`: dimensions come after the pointers.

```cpp
#include "matrix_pro/core/heterogeneous_scheduler.hpp"

void split_product(const float* a, int lda, const float* b, int ldb, float* c, int ldc,
                   int m, int n, int k) {
    matrix_pro::HeterogeneousScheduler::instance().gemm(
        a, lda, b, ldb, c, ldc, m, n, k, 1.f, 0.f);
}
```

Rows are split. The GPU share runs on a CUDA stream through `hw::cuda_gemm` (cuBLAS). The CPU share runs `gemm_f32` on a helper thread and the work-stealing pool. The two sides write disjoint C rows, so the join does not copy. If B’s host bytes match the previous call, the device copy of B is skipped. The split uses measured GFLOPS and the PCIe cost. When both devices exist, at least one tile stays on the CPU and one on the GPU.

`plan(m, n, k)` returns a `GemmSplit` (`cpu_rows`, `gpu_rows`, `cpu_gflops`, `gpu_gflops`, `pcie_gbps`). `calibrate()` fills those rates. This scheduler is the current CPU+GPU row split. It is not the deleted hybrid GEMM experiment at `N=512`.

## 5. conv2d CUDA kernel

Public signature in `include/matrix_pro/nn/conv.hpp`:

```cpp
Tensor conv2d(const Tensor& input, const Tensor& weights, const Tensor& bias,
              std::size_t stride = 1, std::size_t padding = 0);
```

`input` is NCHW `(N, C, H, W)`. `weights` is `(K, C, kH, kW)`. `bias` is `(K)`. The op is cross-correlation (the kernel is not flipped). Output height is `(H + 2*padding - kH) / stride + 1`, and the same for width.

With cuDNN force-disabled, `MATRIX_PRO_HAS_CUDNN` is unset and `src/nn/conv.cu` launches `conv2d_kernel`. One thread owns one output element, adds `bias[filter]`, and accumulates the input window. The launch uses the compute stream and marks the host mirror stale.

```cpp
#include "matrix_pro/nn/conv.hpp"

matrix_pro::Tensor conv_example(const matrix_pro::Tensor& x,
                                const matrix_pro::Tensor& w,
                                const matrix_pro::Tensor& b) {
    return matrix_pro::conv2d(x, w, b, /*stride=*/1, /*padding=*/1);
}
```

Measured on this machine against Nexus CPU conv. Protocol shape `8×3×32×32`, 16 filters, `3×3`, padding 1:

| Path | GFLOPS |
|---|---:|
| Nexus CPU conv | 6.39 |
| Device-resident CUDA kernel | 48.7 |
| Host round-trip | 19.0 |

These three numbers are not compared with PyTorch.

`max_pool2d` and `avg_pool2d` use `pool2d_kernel` on the same force-disabled build. Their signatures are in the same header. This manual does not assign them a GFLOPS figure.

## 6. Deleted experiments

The following were slower and have been deleted. They are not current kernels and they are not selected by `gemm_f32`:

| Experiment | Status |
|---|---|
| `Nc` tile that repacks A | Deleted |
| MASM 6×16 | Deleted |
| Hybrid GEMM at `N=512` | Deleted |

The kernel that remains is `avx2-4x16`. A comment in `src/hw/gemm_avx2.cpp` records that a 6×16 kernel and a column-sliced L2 schedule were tried and were slower at `N=1024` and `N=2048`; the kept schedule is 4×16 with `kMc = 64` and `kKc = 512`.

## 7. What this manual does not claim

- No world ranking.
- No GFLOPS other than the 2026-10-21 tables above.
- The 1 October 2026 GPU cuBLAS / cuBLASLt tables in `PERFORMANCE_REPORT.md` are that protocol only.
- cuDNN is force-disabled for the measured build. The `#ifdef MATRIX_PRO_HAS_CUDNN` branches are not that build.
- AVX-512 is not selected on this CPU.
