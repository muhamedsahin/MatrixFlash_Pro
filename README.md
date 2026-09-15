# MatrixFlash-Pro

CUDA tabanli, okunabilir bir C++17 matris kutuphanesi. CUDA kernel ayrintilari
`matrix_pro::Matrix` API'sinin arkasinda tutulur.

## Moduler kaynak yapisi

```text
include/matrix_pro/
	matrix.hpp                 Public Matrix API
	operations.hpp             GPU operasyon fonksiyonlari
	cuda_utils.hpp             CUDA hata ve senkronizasyon yardimcilari

src/
	matrix.cu                  Yasam dongusu, kopyalama, host/device bellek
	matrix_factories.cu        zeros, ones, identity, fill
	operations_elementwise.cu  add, subtract, Hadamard, scalar
	operations_matmul.cu       Shared-memory tiled matris carpimi
	operations_transforms.cu   transpose, ReLU, softmax, flatten, slice
	operations_statistics.cu   sum, mean, norm, trace
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

Matrix input{{1.0f, -2.0f}, {3.0f, 4.0f}};
Matrix weights = Matrix::identity(2);
Matrix output = (input * weights).relu().softmax();

output.download();
```

Operasyonlar device bellekte zincirlenir. Host tarafinda okumak gerektiginde
acikca `download()` cagirilarak gereksiz kopyalarin onune gecilir.

## Derleme ve test

```powershell
cmake -S . -B build
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
& .\build\Release\matrix_pro_basic.exe
& .\build\Release\matrix_pro_benchmark.exe 1024
```

CUDA 12.3+ ve Visual Studio CUDA toolchain gerekir. CMake yerel GPU mimarisini
`CMAKE_CUDA_ARCHITECTURES=native` ile hedefler.
