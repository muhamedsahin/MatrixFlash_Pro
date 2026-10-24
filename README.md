<div align="center">

# ⚡ MatrixFlash-Pro

### C++17 matris kütüphanesi — AVX2 host GEMM, cuBLAS/cuBLASLt aygıt GEMM, CUDA conv

*Nexus ekosisteminin hesap tabanı: her `Linear` ve dikkat çarpımı buraya iner.*

[![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C?style=for-the-badge&logo=cplusplus&logoColor=white)](https://en.cppreference.com/w/cpp/17)
[![CMake](https://img.shields.io/badge/CMake-3.24+-064F8C?style=for-the-badge&logo=cmake&logoColor=white)](CMakeLists.txt)
[![Kernel](https://img.shields.io/badge/kernel-avx2--4x16-FF6F00?style=for-the-badge)](include/matrix_pro/cpu/gemm.hpp)
[![CUDA](https://img.shields.io/badge/cuBLAS-cuBLASLt-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](src/hw/cuda_backend.cu)
[![License](https://img.shields.io/badge/License-Apache--2.0-D22128?style=for-the-badge&logo=apache&logoColor=white)](LICENSE)

[Install](#-install--kurulum) • [`gemm_f32`](#-gemm_f32) • [Device GEMM](#-device-gemm-is-cublas) • [Measurements](#-this-machine-evening-of-2026-10-21) • [Manual](#-manual) • [Nexus](https://github.com/muhamedsahin/Nexus)

</div>

> Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.

---

## ✨ At a glance

<table>
<tr>
<td width="33%" valign="top">

### 🖥 Host GEMM
`gemm_f32` — row-major `C = αAB + βC`. On this machine it selects **`avx2-4x16`** (B panel packed once). AVX-512 is not selected on Zen 3.

</td>
<td width="33%" valign="top">

### 🎮 Device GEMM
`matrix_pro::hw::cuda_gemm` issues `cublasSgemm`; `GemmPlan` uses cuBLAS or cuBLASLt. Default precision is strict FP32; TF32 is an explicit option.

</td>
<td width="33%" valign="top">

### 🧩 Beyond GEMM
Thread pool, heterogeneous CPU/GPU scheduler, collectives and a CUDA conv2d kernel (`src/nn/conv.cu`, not cuDNN).

</td>
</tr>
</table>

---

## 🛠 Install / Kurulum

Tested host tools: CMake 3.24+, Visual Studio 2022 (MSVC). CUDA is optional (measurements used CUDA 13.4). CPU GEMM, the thread pool and the scheduler build with CUDA off.

```powershell
# Preset build (CUDA on, if the toolkit is on PATH)
cmake --preset release
cmake --build --preset release
ctest --preset release-tests

# Local performance script: finds vcvars64.bat, configures Ninja, force-disables cuDNN
python tools/build_local.py --reconfigure --jobs 4

# CPU only
cmake -S . -B build-cpu -DMATRIX_PRO_ENABLE_CUDA=OFF
cmake --build build-cpu --config Release
```

Link the `matrix_pro` target. Public CPU headers are under `include/`.

---

## 🧮 `gemm_f32`

Declared in `include/matrix_pro/cpu/gemm.hpp`:

```cpp
void gemm_f32(int m, int n, int k, const float* a, int lda, const float* b, int ldb,
              float* c, int ldc, float alpha, float beta);
```

`A` is `m×k` (stride `lda`), `B` is `k×n` (stride `ldb`), `C` is `m×n` (stride `ldc`).

```cpp
#include "matrix_pro/cpu/gemm.hpp"

#include <vector>

int main() {
    const int m = 2, n = 2, k = 2;
    const std::vector<float> a{1.f, 2.f, 3.f, 4.f};
    const std::vector<float> b{5.f, 6.f, 7.f, 8.f};
    std::vector<float> c(static_cast<std::size_t>(m * n), 0.f);
    matrix_pro::gemm_f32(m, n, k, a.data(), k, b.data(), n, c.data(), n, 1.f, 0.f);
}
```

`gemm_kernel_name()` returns the selected kernel — `avx2-4x16` on this CPU.

---

## 🎮 Device GEMM is cuBLAS

`include/matrix_pro/detail/cuda_hooks.hpp`:

```cpp
namespace matrix_pro::hw {
void cuda_gemm(int device, int m, int n, int k, const float* a, int lda, const float* b, int ldb,
               float* c, int ldc, float alpha, float beta);
}
```

Pointers are device pointers; the body in `src/hw/cuda_backend.cu` calls `cublasSgemm`. Without the CUDA toolkit the same signature throws `MatrixProError`.
`GemmPlan` / `gemm` / `gemm_into` in `include/matrix_pro/ops/gemm.hpp` are the Matrix-level device API.

---

## 📊 This machine, evening of 2026-10-21

AMD Ryzen 7 5800H (no AVX-512). The serial AVX-512 kernel does not distribute rows, so `gemm_f32` keeps AVX2 when AVX2 and FMA are present. **Not a world ranking.**

**Square GEMM, `N=2048`**

| Run | GFLOPS |
| --- | ---: |
| Nexus | 330 |
| Nexus | 362 |
| Nexus | 369 |
| TensorFlow | **389** |

**CUDA conv vs this library's CPU conv** — `8×3×32×32`, 16 filters, `3×3`, padding 1:

| Path | GFLOPS |
| --- | ---: |
| Nexus CPU conv | 6.39 |
| Device-resident CUDA kernel | **48.7** |
| Host round-trip | 19.0 |

That CUDA figure is not a comparison with PyTorch. cuDNN is force-disabled in the measured build.

> 📈 Nexus FAZ 0–3 audit (2026-10-24), 15-run median of the same protocol: GEMM N=2048 **391 GFLOPS** (+8.1% vs the Oct 21 median).

<details>
<summary><b>Deleted experiments (slower, not the current kernel)</b></summary>

- an `Nc` tile that repacks A
- the MASM 6×16 kernel
- hybrid GEMM at `N=512`

The 1 October 2026 GPU tables in `PERFORMANCE_REPORT.md` are an older cuBLAS protocol on an RTX 3070 Laptop; they are not this CPU kernel and not a ranking of libraries.

</details>

---

## 📖 Manual

`doc/manual.md` is the technical manual source (install, GEMM, cuBLAS, heterogeneous scheduler, conv2d CUDA kernel, deleted experiments). Pandoc is not installed; the existing PDFs were not regenerated:

- `MatrixFlash_Pro_Technical_Manual.pdf`
- `doc/public/MatrixFlash_Pro_Technical_Manual.pdf`

## 📄 License

Apache-2.0 — the license text is the [`LICENSE`](LICENSE) file in this directory.

---

<div align="center">

**🎓 Bu proje BTÜ (Bursa Teknik Üniversitesi) Bilgisayar Mühendisliği 1. sınıf öğrencisi tarafından yapay zeka destekli olarak yapıldı.**

<sub>Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.</sub>

<sub>⚡ [Nexus](https://github.com/muhamedsahin/Nexus) ekosisteminin bir parçasıdır.</sub>

</div>
