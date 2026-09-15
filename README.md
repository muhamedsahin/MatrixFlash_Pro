# MatrixFlash-Pro

CUDA tabanli, okunabilir bir C++17 matris kutuphanesi. CUDA kernel ayrintilari
`matrix_pro::Matrix` API'sinin arkasinda tutulur. Performans odakli: matris
carpimi **cuBLAS** ile yapilir, indirgemeler ve aktivasyonlar GPU'da calisir.

## Ozellikler

- **cuBLAS matris carpimi** (row-major uyumlu) — elle yazilmis kernel'dan cok daha hizli
- **GPU indirgemeleri**: `sum, mean, min, max, argmin, argmax, variance, stddev, l1_norm, l2_norm, abs_max`
- **Eksen indirgemeleri**: `row_sum, col_sum`
- **Elementwise matematik**: `exp, log, sqrt, abs, clamp, pow, add_scalar, sigmoid, tanh, negatif`
- **Yayin (broadcast)**: `add_row_vector, add_col_vector, multiply_row_vector, multiply_col_vector`
- **Diger carpimlar**: `outer_product`
- **Aktivasyonlar**: `relu, softmax` (cok bloklu, kararli), `sigmoid, tanh`
- **Dogusturucular**: `zeros, ones, identity, random, uniform, randn, glorot (Xavier)`
- **Kalici veri**: `save("dosya")` / `Matrix::load("dosya")`
- **Matris dogrusal cebir**: `transpose, flatten, slice, trace, determinant, inverse`

## Moduler kaynak yapisi

```text
include/matrix_pro/
	matrix.hpp                 Public Matrix API
	operations.hpp             GPU operasyon fonksiyonlari
	cuda_utils.hpp             CUDA hata ve senkronizasyon yardimcilari

src/
	matrix.cu                  Yasam dongusu, kopyalama, save/load
	matrix_factories.cu        zeros, ones, identity, random/randn/glorot
	operations_elementwise.cu  add, subtract, Hadamard, scalar, matematik, broadcast
	operations_matmul.cu       cuBLAS matris carpimi + outer_product
	operations_transforms.cu   transpose, relu, softmax, flatten, slice
	operations_statistics.cu   GPU indirgemeleri (sum/mean/min/max/...)
	operations_advanced.cu     determinant ve inverse
	cuda_utils.cu              CUDA hata kontrolu

tests/                       Davranis testleri
benchmarks/perf_matrix.cpp   GPU matmul benchmark'i
examples/basic_gpu_usage.cpp Temel kullanim
```

## Kullanim

```cpp
#include "matrix_pro/matrix.hpp"

using matrix_pro::Matrix;

// onceki ornek
Matrix input{{1.0f, -2.0f}, {3.0f, 4.0f}};
Matrix weights = Matrix::identity(2);
Matrix output = (input * weights).relu().softmax();
output.download();

// yeni ozellikler
Matrix a = Matrix::random(256, 128);
Matrix b = Matrix::randn(128, 64);
Matrix c = a * b;                      // cuBLAS
Matrix bias{{0.1f, 0.2f, 0.3f}};      // satir vektoru (len == cols)
Matrix z = c.add_row_vector(bias).sigmoid();
z.download();

float sum = z.sum();                   // GPU indirgemesi
std::size_t best = z.argmax();         // en buyuk elemanin indeksi
float stdv = z.stddev();

Matrix w = Matrix::uniform(4, 4, -1.0f, 1.0f);
Matrix inv = w.inverse();              // tersi
w.save("w.bin");
Matrix reloaded = Matrix::load("w.bin");
```

Operasyonlar device bellekte zincirlenir. Host tarafinda okumak gerektiginde
acikca `download()` cagirilir.

## Derleme ve test

```powershell
cmake -S . -B build
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
& .\build\Release\matrix_pro_basic.exe
& .\build\Release\matrix_pro_benchmark.exe 1024
```

CUDA 12.3+ (cuBLAS dahil) ve Visual Studio CUDA toolchain gerekir. CMake yerel
GPU mimarisini `CMAKE_CUDA_ARCHITECTURES=native` ile hedefler. `find_package(CUDAToolkit)`
cuBLAS'i bulur.
