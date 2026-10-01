<div align="center">

# ⚡ MatrixFlash-Pro

### CUDA Tabanlı, Yüksek Performanslı C++17 Matris Kütüphanesi

*cuBLAS destekli matris çarpımı, GPU üzerinde çalışan indirgemeler ve aktivasyon fonksiyonları*

[![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?style=flat&logo=c%2B%2B&logoColor=white)](https://en.cppreference.com/w/cpp/17)
[![CUDA](https://img.shields.io/badge/CUDA-12.3%2B-76B900?style=flat&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![CMake](https://img.shields.io/badge/CMake-Build-064F8C?style=flat&logo=cmake&logoColor=white)](https://cmake.org/)
[![Platform](https://img.shields.io/badge/Platform-Windows-0078D6?style=flat&logo=windows&logoColor=white)](#)
[![License](https://img.shields.io/badge/License-See%20LICENSE-lightgrey?style=flat)](./LICENSE)

</div>

---

## 📖 İçindekiler

- [Proje Hakkında](#-proje-hakkında)
- [Öne Çıkan Özellikler](#-öne-çıkan-özellikler)
- [Neden MatrixFlash-Pro?](#-neden-matrixflash-pro)
- [Mimari ve Kaynak Kod Yapısı](#-mimari-ve-kaynak-kod-yapısı)
- [Gereksinimler](#-gereksinimler)
- [Kurulum ve Derleme](#-kurulum-ve-derleme)
- [Hızlı Başlangıç](#-hızlı-başlangıç)
- [API Referansı](#-api-referansı)
- [Testler](#-testler)
- [Performans / Benchmark](#-performans--benchmark)
- [Yol Haritası](#-yol-haritası)
- [Katkıda Bulunma](#-katkıda-bulunma)
- [Lisans](#-lisans)
- [Geliştirici Hakkında](#-geliştirici-hakkında)

---

## 🧩 Proje Hakkında

**MatrixFlash-Pro**, NVIDIA GPU'lar üzerinde çalışan, okunabilir ve modern **C++17** ile yazılmış bir matris işlem kütüphanesidir. Kütüphanenin temel felsefesi; CUDA kernel karmaşıklığını kullanıcıdan tamamen soyutlayarak, sade ve akıcı bir `matrix_pro::Matrix` arayüzü sunmaktır.

Performans kritik matris çarpımı işlemleri, elle yazılmış CUDA kernel'ları yerine NVIDIA'nın optimize edilmiş **cuBLAS** kütüphanesi üzerinden gerçekleştirilir. Buna karşın indirgeme (reduction) işlemleri, elementwise matematik operasyonları ve aktivasyon fonksiyonları özel olarak yazılmış GPU kernel'ları ile çalışır. Bu hibrit yaklaşım, hem endüstri standardı performansı hem de esnekliği bir arada sunar.

Son geliştirme turunda kütüphane, yüksek performans için aşağıdaki optimizasyonları aldı:

- **Shape-aware GEMM dispatcher** (`src/ops/gemm/`): mikro kernel, GEMV, algo-cache’li cublasLt, TENSOR_OP GemmEx
- **`multiply_into` + `device_only` çıktı** — tahsis maliyetini hot path’ten çıkarır
- **`gemm_bias_relu` / `gemm_bias_gelu`** — cache'li cuBLASLt fused epilogue
- **TF32 Tensor Core** matematik modu + 64 MiB kalıcı cuBLAS workspace
- GPU bellek havuzu, pinned host, async transferler
- float4 vektörize broadcast / reduction

1 Ekim 2026 ölçümü (RTX 3070 Laptop): uzun vektör × matris yolunda önceki sürüme göre **3,98×** hızlanma. Büyük kare matrislerde evrensel üstünlük yok; ölçülen gerilemeler de [detaylı raporda](./PERFORMANCE_REPORT.md) yer alıyor. Ham cuBLAS, cuBLASLt ve NumPy/OpenBLAS ölçümleri ile tüm örnekler repoda saklanıyor.

Kütüphane özellikle şu kullanım senaryoları için tasarlanmıştır:

- 🧠 Küçük/orta ölçekli sinir ağı katmanlarının GPU üzerinde manuel olarak prototiplenmesi
- 📊 Bilimsel hesaplama ve doğrusal cebir işlemlerinin hızlandırılması
- 🎓 CUDA/GPU programlamayı öğrenmek isteyenler için okunabilir, modüler bir referans kod tabanı
- ⚙️ cuBLAS entegrasyonunun gerçek bir C++ projesinde nasıl yapılandırılacağının örneklenmesi

---

## ✨ Öne Çıkan Özellikler

### 🔢 Temel Matris İşlemleri
| Kategori | İşlemler |
|---|---|
| **Matris Çarpımı** | Shape-aware GEMM: mikro / GEMV / cublasLt (algo-cache) / TENSOR_OP cuBLAS |
| **Hot path** | `multiply_into`, `gemm_bias_relu`, `gemm_bias_gelu` |
| **Doğrusal Cebir** | `transpose`, `flatten`, `slice`, `trace`, `determinant`, `inverse`, QR/SVD/eig |
| **Dış Çarpım** | `outer_product` |

### ⚙️ Tekrar Kullanılabilir GEMM ve LU Planları

- `GemmPlan`: A/B transpose, alpha/beta accumulation, açık FP32/TF32,
  fused bias/ReLU/GELU, en fazla 16 adayla isteğe bağlı tuning.
- `gemm` / `gemm_into`: bounded thread/device/stream plan cache'i.
- `LUFactorization`: bir kez faktörleme, preallocated `solve_into`, çoklu RHS,
  transpose çözümü ve işaretli `slogdet`.
- `matrix_exp`: [13/13] Padé scaling/squaring; sayısal iç çarpımlar strict FP32.
- Plan/buffer'lar capture öncesinde hazırlanır; sıcak yürütme descriptor,
  heuristic veya çıktı tahsisi yapmaz. Planlar oluşturuldukları thread/device/stream'e bağlıdır.

```cpp
#include "matrix_pro/matrix_pro.hpp"
using namespace matrix_pro;

Matrix a = Matrix::random(512, 512, 17);
Matrix b = Matrix::random(512, 512, 23);
Matrix bias = Matrix::ones(1, 512);
Matrix c(512, 512, MemoryMode::device_only);
GemmOptions options;
options.precision = GemmPrecision::tf32; // hız/doğruluk tercihi açık
options.epilogue = GemmEpilogue::bias_relu;
GemmPlan plan(512, 512, 512, options);
plan.tune(a, b, c, 1, 0, &bias);         // isteğe bağlı; senkronize eder
plan.execute(a, b, c, 1, 0, &bias);
c.download();

Matrix coefficients(2, 2, {4, 1, 1, 3});
Matrix rhs(2, 1, {1, 2});
LUFactorization factor(coefficients);
Matrix solution = factor.solve(rhs);    // aynı factor diğer RHS'lerde tekrar kullanılır
auto signed_log = factor.slogdet();
```

Yeni `gemm` API'sinin varsayılanı strict FP32'dir; eski `multiply_into` API'si
mevcut şekle bağlı precision politikasını korur. Tuning her şekli hızlandırmaz.
Graph ve buffer ömrü sözleşmesi için [rapora](./PERFORMANCE_REPORT.md#8-kullanım-sözleşmesi-ve-sınırlar),
çalışır graph örneği için [`examples/planned_gemm.cpp`](./examples/planned_gemm.cpp) dosyasına bakın.

### 📉 GPU Üzerinde İndirgemeler (Reductions)
```
sum · mean · min · max · argmin · argmax · variance · stddev · l1_norm · l2_norm · abs_max
```
Ayrıca eksen bazlı indirgemeler: `row_sum`, `col_sum`

### 🧮 Elementwise Matematiksel Operasyonlar
```
exp · log · sqrt · abs · clamp · pow · add_scalar · negate · divide · sigmoid · tanh
```
Elementwise bölme hem matris-matris (`A / B`) hem skaler (`A / 2.0f`) biçiminde desteklenir.

### 📡 Yayınlama (Broadcasting)
- `add_row_vector` — satır vektörünü tüm satırlara yayma
- `add_col_vector` — sütun vektörünü tüm sütunlara yayma
- `multiply_row_vector` / `multiply_col_vector`
- `broadcast_add` / `broadcast_multiply` — genel 2B yayınlama; `(N, 1) ⊗ (1, M)` dahil her uyumlu şekil kombinasyonu

### 🛡️ Hata Yönetimi, Bellek Modu ve Cihaz Kontrolü
- **Tek hata tabanı:** kütüphanenin fırlattığı tüm istisnalar `matrix_pro::MatrixProError` (→ `std::runtime_error`) tabanından türer. Alt türler: `ShapeMismatchError`, `InvalidArgumentError`, `OutOfRangeError`, `CudaError`, `SolverError`, `IoError`. Böylece tek bir `catch (const matrix_pro::MatrixProError&)` ile tüm kütüphane hataları yakalanır, istenirse nedene özel türle ayrıştırılır.
- **Cihaz seçimi (çoklu-GPU):** `device_count()`, `current_device()`, `set_device(int)` ve RAII `DeviceScope`. Bellek havuzu cihaz-etiketli olduğundan bloklar yanlış cihaza geri dönmez.
- **Thread-safe yürütme:** her `(host thread, cihaz)` çifti kendi CUDA stream'i ve cuBLAS/cuSOLVER/cuDNN handle'ına sahiptir; bağımsız CPU thread'leri paylaşılan tek bir stream'e takılmaz.
- **Device-resident mod:** `MemoryMode::device_only` ile host kopyası olmadan çalışılır; host tamponu yalnızca `download()` çağrıldığında talep üzerine oluşturulur (RAM tasarrufu, gereksiz kopyalama yok).

```cpp
#include "matrix_pro/core/matrix.hpp"

matrix_pro::Matrix w(1024, 1024, matrix_pro::MemoryMode::device_only); // host kopyası yok
w.fill(0.0f);                       // doğrudan cihaz üzerinde yaz
matrix_pro::Matrix y = w + w;       // hesap GPU'da kalır
y.download();                       // sadece gerekince host'a indir

for (int d = 0; d < matrix_pro::device_count(); ++d) {
    matrix_pro::DeviceScope scope(d); // kapsam sonunda önceki cihaza döner
    // ... cihaz d üzerinde iş ...
}
```

### 🔄 Host/Device Senkronizasyon Sözleşmesi (stale-mirror)
GPU yazan **her** operasyon sonucu yalnızca device tamponuna yazar ve çıktının host aynasını
"bayat" (`host_current() == false`) olarak işaretler. Kütüphane bu nedenle **asla sessizce bayat
(sıfır) host verisi döndürmez**:

- `at()` ve `const data()` bayat aynada **`InvalidArgumentError` fırlatır**; `download()` aynayı
  tazeler ve sonrasında okuma güvenlidir. (`host_current()` ile bayatlığı sorgulayabilirsiniz.)
- `device_only` modunda host aynası yoktur: `download()` çağrılana kadar `at()`/`data()` exception fırlatır.
- Konvansiyonel kural: host tarafında değer değiştirdikten sonra (`at()`, non-const `data()`) cihaz
  tarafını güncellemek için `upload()` çağırın; cihaz tarafı işlemler (`+`, `relu`, `matmul`, ...)
  sonucu okumak için `download()` çağırın.

```cpp
Matrix a = Matrix::ones(2, 2);
Matrix c = a + a;        // GPU'da hesaplandı; host aynası bayat
// c.at(0, 0);           // ✗ InvalidArgumentError: host mirror is stale
c.download();            // ✓ aynayı tazele
float v = c.at(0, 0);    // 2.0
```
Bu sözleşme `Tensor` için de aynıdır; `reshape`/`concat`/`stack` gibi şekil operasyonları da her iki
eksen için aynı politikayı izler.

### 🎯 Aktivasyon Fonksiyonları
- `relu`, `softmax` — çok bloklu ve **sayısal olarak kararlı** implementasyon
- `sigmoid`, `tanh`, `leaky_relu`, `elu`, `gelu`, `swish`
- `softplus`, `mish`, `hardtanh`, `hardsigmoid`, `hardswish`, `selu`
- `prelu` — `1×1` veya girdiyle aynı şekilli öğrenilebilir `alpha` parametresi

### 🧠 Makine Öğrenmesi Katmanı
- `Variable` (Matrix) ve `VarTensor` (ND Tensor) ile GPU üzerinde otomatik türev ve geri yayılım grafiği
- `batch_norm`, `layer_norm` ve inverted `dropout`
- NCHW `conv2d`, `max_pool2d`, `avg_pool2d` — **forward ve backward** (im2col tabanlı, cuBLAS destekli)
- cuDNN kuruluysa conv/pooling için cuDNN backend’i; yoksa CUDA kernel fallback’i
- `Tensor` değerleri device belleğinde tutulur; yalnızca açık `download()` çağrısı host’a veri taşır

### 🔁 Otomatik Türev (Autograd)
`Variable` (Matrix) ve `VarTensor` (rank-N Tensor) için tam geri yayılım desteği.
Graf, topolojik sıralı bir tape üzerinde çalışır ve tüm ara değerler GPU’da kalır.

| Grup | Geri yayılımı desteklenen işlemler |
|---|---|
| **Aritmetik** | `add`, `subtract`, `elementwise_multiply`, `divide`, `matmul`, `multiply(scalar)`, `add_scalar`, `negate`, `pow` |
| **Broadcast** | `add_row_vector`, `add_col_vector`, `multiply_row_vector`, `multiply_col_vector` (backward’da eksen boyunca sum-reduce) |
| **Şekil** | `transpose`, `flatten`, `reshape`, `slice` (scatter-grad kernel’i ile) |
| **Aktivasyon** | `relu`, `leaky_relu`, `elu`, `gelu`, `swish`, `sigmoid`, `tanh`, `softplus`, `mish`, `hardtanh`, `hardsigmoid`, `hardswish`, `selu`, `prelu` |
| **Tensor/CNN** | `batch_matmul` (strided-batched GEMM), `conv2d` (girdi + ağırlık + bias), `max_pool2d` (argmax maskesi), `avg_pool2d` |

### 🧩 Özel Operasyon (Custom Op) Altyapısı
Kullanıcı, CUDA kernel yazmadan mevcut primitifleri kompoze ederek kendi fonksiyonunu tanımlayabilir:

```cpp
// Genel eleman-bazlı fonksiyon: forward + backward closure
Variable custom_unary(const std::string& name,
                      std::function<Matrix(const Matrix&)> forward,
                      std::function<Matrix(const Matrix& grad_out, const Matrix& input, const Matrix& output)> backward) const;

// Genel kayıp fonksiyonu: skaler loss döndürür, pred/target gradyanlarını üretir
Variable custom_loss(const Variable& prediction, const Variable& target,
                     std::function<float(const Matrix&, const Matrix&)> forward,
                     std::function<Matrix(const Matrix&, const Matrix&)> backward);
```

`VarTensor` için de aynı `custom_unary` / `custom_loss` ikilisi (Tensor imzalarıyla) mevcuttur.
Kendi CUDA kernel’inizi yazmak isterseniz, `Matrix -> Matrix` imzalı bir fonksiyonu `forward`/`backward`
olarak vermeniz yeterlidir; ayrı bir plugin API’si gerekmez.

### 📉 Kayıp (Loss) Fonksiyonları
| Fonksiyon | Not |
|---|---|
| `mse_loss` | Ortalama karesel hata |
| `mae_loss` | Ortalama mutlak hata; `\|d\| = 0` noktasında subgradient `0` |
| `huber_loss(pred, target, delta)` | Smooth L1 |
| `bce_with_logits_loss` | **Fused** sigmoid + BCE; `log(0)` patlamasına karşı kararlı |
| `softmax_cross_entropy_loss` | **Fused** softmax + CE; tek geçişte skaler + gradyan |
| `kl_divergence_loss` | `mean(t · (log t − log p))` |

Tümü `Variable` (ve `custom_loss` üzerinden `VarTensor`) seviyesinde `1×1` skaler döndürür ve
doğrudan `.backward()` ile zincirlenebilir.

### 🧱 Şekil ve Yardımcı Operasyonlar
- `reshape(rows, cols)` — genel yeniden şekillendirme (`flatten` bunun özel durumu)
- `concat({A, B, ...}, axis)` — satır (0) veya sütun (1) ekseninde birleştirme
- `stack({A, B, ...})` — matrisleri `(n, r*c)` biçiminde üst üste yığma
- `one_hot(indices, classes)` — GPU üzerinde tek-sıcak kodlama
- `slice_scatter(grad, like, ...)` — slice geri yayılımı için scatter kernel’i

### 🎲 Matris Oluşturucular
```
zeros · ones · identity · random · uniform · randn · glorot (Xavier init)
```

### 💾 Kalıcı Veri Desteği
- `matrix.save("dosya")` ile matrisi diske yazma
- `Matrix::load("dosya")` ile diskten geri yükleme
- Dosya biçimi **magic number (`MFMP`) + sürüm alanı** ile sürümlenmiştir; sürümsüz (legacy) eski
  dosyalar geriye dönük olarak okunabilir

---

## 🚀 Neden MatrixFlash-Pro?

- **Performans önceliği:** Matris çarpımı gibi maliyetli işlemler, elle yazılmış naif kernel'lardan çok daha hızlı çalışan cuBLAS üzerinden yürütülür. Son versiyonda TF32 moduyla Tensor Core destekleri de devreye alındı.
- **Zincirleme (chaining) API:** İşlemler device belleğinde art arda zincirlenebilir; host tarafına veri yalnızca açıkça `download()` çağrıldığında aktarılır. Bu, gereksiz host↔device veri transferini ortadan kaldırarak performans kaybını önler.
- **Bellek-odaklı optimizasyon:** Aynı boyuttaki iş parçaları için bellek havuzu, pinned host bellek ve async transfer kullanımı sayesinde sık tekrar eden alloc/free ve veri taşıma yükü azaltıldı.
- **Modüler kod tabanı:** Her operasyon grubu kendi `.cu` dosyasında ayrıştırılmıştır — okunabilirlik ve bakım kolaylığı ön plandadır.
- **Sayısal kararlılık:** `softmax` gibi hassas fonksiyonlar taşma (overflow) sorunlarına karşı kararlı şekilde implemente edilmiştir.
- **Kapsamlı test ve benchmark altyapısı:** Davranış testleri ve GPU matmul benchmark aracı proje ile birlikte gelir; son ölçümler doğrulanmıştır.

---

## 🏗️ Mimari ve Kaynak Kod Yapısı

Kamusal başlıklar `include/matrix_pro/` altındadır. Tek giriş `matrix_pro/matrix_pro.hpp`.
Gövdeler `src/` altındadır. Her kod klasöründe `detail.txt` o klasördeki dosyaları tek tek anlatır.
GEMM alt sistemi için uzun metin: [`src/ops/gemm/detail.txt`](src/ops/gemm/detail.txt).
Kısa harita: [`include/matrix_pro/detail.txt`](include/matrix_pro/detail.txt) ve [`src/detail.txt`](src/detail.txt).

```text
include/matrix_pro/
  matrix_pro.hpp          şemsiye başlık
  core/                   Matrix, Tensor, hata, havuz, cihaz, serileştirme
  ops/gemm.hpp            GemmPlan, gemm, gemm_into
  ops/factorization.hpp   LUFactorization, slogdet
  ops/linalg*.hpp         cuSOLVER ve matrix_exp / trsm
  nn/  autograd/  sparse/ indexing/ view/ rng/ streams/
  detail/gemm/            dispatcher eşikleri; istemci include etmez

src/
  core/  ops/  ops/gemm/  nn/  autograd/  sparse/  indexing/  rng/  streams/
```

`src/ops/gemm/` şekle göre micro, GEMV, tiled, cuBLAS GemmEx ve algo-önbellekli cuBLASLt yollarını tutar.
Eski `multiply` bu dispatcher'a gider. Yeni `GemmPlan` hassasiyeti (FP32 veya TF32), transpose'u ve fused epilogue'u açık seçer.

---

## 📋 Gereksinimler

Projeyi derlemeden önce sisteminizde aşağıdakilerin kurulu olduğundan emin olun:

| Bileşen | Minimum Sürüm | Not |
|---|---|---|
| **CUDA Toolkit** | 12.3+ | cuBLAS dahil olmalıdır |
| **Visual Studio** | CUDA destekli bir sürüm | Windows CUDA toolchain için gereklidir |
| **CMake** | Güncel bir sürüm | `find_package(CUDAToolkit)` ile cuBLAS otomatik bulunur |
| **Uyumlu NVIDIA GPU** | CUDA destekli | `CMAKE_CUDA_ARCHITECTURES=native` ile yerel mimari otomatik hedeflenir |

> 💡 **Not:** CUDA kurulumunda sorun yaşıyorsanız, NVIDIA'nın **local installer** (network installer değil) sürümünü tercih etmeniz tavsiye edilir; bu sürüm indirme sırasında takılma sorunlarını büyük ölçüde azaltır.

---

## ⚙️ Kurulum ve Derleme

Depoyu klonladıktan sonra aşağıdaki adımları izleyin:

```powershell
# 1. Build dizinini oluşturun ve CMake yapılandırmasını yapın
cmake -S . -B build

# 2. Release modunda derleyin
cmake --build build --config Release --parallel

# 3. Testleri çalıştırın
ctest --test-dir build -C Release --output-on-failure

# 4. Proje kök dizininden örnek uygulamayı çalıştırın
.\build\Release\matrix_pro_basic.exe

# 5. Benchmark aracını çalıştırın (1024x1024 boyutunda)
.\build\Release\matrix_pro_benchmark.exe 1024
```

---

## 🏁 Hızlı Başlangıç

Aşağıdaki örnek, temel kullanımdan ileri düzey özelliklere kadar kütüphanenin API'sini göstermektedir:

```cpp
#include "matrix_pro/core/matrix.hpp"

using matrix_pro::Matrix;

int main() {
    // --- Temel kullanım ---
    Matrix input{{1.0f, -2.0f}, {3.0f, 4.0f}};
    Matrix weights = Matrix::identity(2);
    Matrix output = (input * weights).relu().softmax();
    output.download();

    // --- Rastgele matrisler ve cuBLAS çarpımı ---
    Matrix a = Matrix::random(256, 128);
    Matrix b = Matrix::randn(128, 64);
    Matrix c = a * b; // cuBLAS ile hızlandırılmış çarpım

    // --- Broadcasting ---
    Matrix bias{{0.1f, 0.2f, 0.3f}}; // satır vektörü (uzunluk == sütun sayısı)
    Matrix z = c.add_row_vector(bias).sigmoid();
    z.download();

    // --- GPU indirgemeleri ---
    float sum   = z.sum();
    std::size_t best = z.argmax(); // en büyük elemanın indeksi
    float stdv  = z.stddev();

    // --- Doğrusal cebir ---
    Matrix w   = Matrix::uniform(4, 4, -1.0f, 1.0f);
    Matrix inv = w.inverse();

    // --- Kalıcı veri (save / load) ---
    w.save("w.bin");
    Matrix reloaded = Matrix::load("w.bin");

    return 0;
}
```

> ⚠️ **Önemli:** Operasyonlar device (GPU) belleğinde zincirlenir. Sonuçları host (CPU) tarafında okumanız gerektiğinde mutlaka açıkça `download()` metodunu çağırmalısınız.

---

## 📚 API Referansı

<details>
<summary><b>🔹 Oluşturucular (Factories)</b></summary>

| Fonksiyon | Açıklama |
|---|---|
| `Matrix::zeros(rows, cols)` | Sıfırlardan oluşan matris |
| `Matrix::ones(rows, cols)` | Birlerden oluşan matris |
| `Matrix::identity(n)` | n×n birim matris |
| `Matrix::random(rows, cols)` | Rastgele değerli matris |
| `Matrix::uniform(rows, cols, min, max)` | Belirli aralıkta düzgün dağılım |
| `Matrix::randn(rows, cols)` | Normal (Gauss) dağılımlı rastgele değerler |
| `Matrix::glorot(rows, cols)` | Xavier/Glorot ağırlık başlatma |

</details>

<details>
<summary><b>🔹 Elementwise ve Skaler İşlemler</b></summary>

`exp`, `log`, `sqrt`, `abs`, `clamp`, `pow`, `add_scalar`, `negate`, bölme (`/` matris ve skaler), toplama/çıkarma, Hadamard (elementwise) çarpım

</details>

<details>
<summary><b>🔹 Yayınlama (Broadcasting)</b></summary>

`add_row_vector`, `add_col_vector`, `multiply_row_vector`, `multiply_col_vector`, `broadcast_add`, `broadcast_multiply`

</details>

<details>
<summary><b>🔹 İstatistik / İndirgemeler</b></summary>

`sum`, `mean`, `min`, `max`, `argmin`, `argmax`, `variance`, `stddev`, `l1_norm`, `l2_norm`, `abs_max`, `row_sum`, `col_sum`

</details>

<details>
<summary><b>🔹 Dönüşümler, Şekil ve Aktivasyonlar</b></summary>

`transpose`, `flatten`, `reshape`, `slice`, `concat`, `stack`, `one_hot`, `relu`, `softmax`, `sigmoid`, `tanh`, `leaky_relu`, `elu`, `gelu`, `swish`, `softplus`, `mish`, `hardtanh`, `hardsigmoid`, `hardswish`, `selu`, `prelu`

</details>

<details>
<summary><b>🔹 Otomatik Türev (Variable / VarTensor)</b></summary>

`Variable` (Matrix) ve `VarTensor` (ND Tensor): `value`, `grad`, `requires_grad`, `zero_grad`, `backward`

Geri yayılımı desteklenen işlemler: `add`, `subtract`, `elementwise_multiply`, `divide`, `matmul`, `batch_matmul`, `multiply`, `add_scalar`, `negate`, `pow`, `add_row_vector`, `add_col_vector`, `multiply_row_vector`, `multiply_col_vector`, `transpose`, `flatten`, `reshape`, `slice`, tüm aktivasyonlar, `conv2d`, `max_pool2d`, `avg_pool2d`

Özel operasyonlar: `custom_unary(name, forward, backward)`, `custom_loss(prediction, target, forward, backward)`

</details>

<details>
<summary><b>🔹 Kayıp (Loss) Fonksiyonları</b></summary>

`mse_loss`, `mae_loss`, `huber_loss`, `bce_with_logits_loss`, `softmax_cross_entropy_loss`, `kl_divergence_loss`

</details>

<details>
<summary><b>🔹 Makine Öğrenmesi / CNN</b></summary>

`Variable`, `VarTensor`, `backward`, `batch_norm`, `layer_norm`, `dropout`, `conv2d`, `max_pool2d`, `avg_pool2d`

`conv2d` ve pooling girdileri NCHW düzeninde Tensor bekler. cuDNN bulunamazsa aynı API otomatik olarak CUDA kernel backend’ine düşer. Tüm CNN operatörleri geri yayılım destekler.

</details>

<details>
<summary><b>🔹 İleri Düzey Doğrusal Cebir</b></summary>

`trace`, `determinant`, `inverse`, `outer_product`

</details>

<details>
<summary><b>🔹 Kalıcılık (Persistence)</b></summary>

`matrix.save("dosya_yolu")` / `Matrix::load("dosya_yolu")` — `MFMP` magic number + sürüm alanlı biçim; legacy dosyalar geriye dönük okunur.

</details>

---

## 🧪 Testler

Proje, `tests/` klasörü altında davranış (behavior) testleri içerir. Bu testler, her operasyonun beklenen matematiksel çıktıyı ürettiğini doğrulamak için CMake/CTest altyapısı ile entegre edilmiştir:

```powershell
ctest --test-dir build -C Release --output-on-failure
```

---

## 📈 Performans / Benchmark

**1 Ekim 2026, RTX 3070 Laptop:** iki ayrı tur, 14 matris şekli, 30 örnek ve
32-GEMM batch. Girdiler sıfır olmayan seeded rastgele FP32 değerler; sonuçlar
bağımsız CPU double dot-product ile kontrol edilir. Tahsis/transfer/tuning zamanı
sıcak GEMM ölçümünün dışındadır. FP32 ve TF32 sonuçları ayrı karşılaştırılır.

| Ölçülen durum | Sonuç |
|---|---|
| `1×4096` · `4096×1024`, eski → yeni default | 185.824 → 46.656 µs; **3.98×** |
| `1024×512` · `512×64`, eski → açık TF32 tuned | 39.264 → 16.304 µs; **2.41×** |
| `1024²`, eski → yeni default | 159.376 → 187.584 µs; yaklaşık %17.7 daha yavaş |
| `2048²`, eski → yeni default | 1164.528 → 1288.800 µs; yaklaşık %10.7 daha yavaş |
| Doğruluk | 26/26 CTest; üç yeni yolda memcheck 0 hata |

Büyük karelerde aynı turdaki cuBLAS/cuBLASLt ile yakın sonuçlar görülür;
“her yerde en hızlı” iddiası yapılmaz. Küçük NumPy/CPU vakaları GPU gönderim
gecikmesinden daha hızlı olabilir. Laptop saatleri ve Windows scheduler'ı
kilitlenmediği için küçük farklar üstünlük kanıtı sayılmaz.

[**Detaylı rapor**](./PERFORMANCE_REPORT.md): bütün şekiller, strict FP32/TF32
vendor karşılaştırması, aynı kapsamlı CUDA Graph tablosu, CPU bağlamı,
gerilemeler, API sınırları ve yeniden üretim komutları.
[**Ham JSON ve test kanıtları**](./benchmarks/reports/2026-10-01/).
PyTorch/CuPy bu ölçüm ortamında kurulu değildi; onlar için sonuç uydurulmadı.

```powershell
python tools/build_local.py --reconfigure --jobs 4
ctest --test-dir build/perf-release -C Release --output-on-failure -j 1
.\build\perf-release\benchmarks\matrix_pro_bench_gemm_fair.exe --output benchmarks/results/run1.json --tag reproduced --warmup 10 --repeats 30 --batch 32
python tools/bench_rivals_fair.py --output benchmarks/results/rivals_fair.json
```

---

## ✅ Son Durum

MatrixFlash-Pro şu anda aşağıdaki temel optimizasyonları içermektedir:

- **TF32 Tensor Core matematik modu**
- gereksiz host→device upload kaldırıldı
- sıfır matrisler için doğrudan `cudaMemset`
- bellek havuzu boyut-sınıfı (power-of-two size-class) bucketing ile alloc/free maliyeti azaltıldı
- pinned host memory ve async transferler
- stream-safe reduction yapısı (`abs_max` dahil tüm indirgemeler tek GPU geçişi)
- cuSOLVER tabanlı ileri düzey doğrusal cebir destekleri
- GPU üzerinde batch matmul ve ND Tensor desteği
- FP16 Tensor Core ve double-accumulation matmul yolları
- GPU compact mask filtreleme ve atomik `any/all`
- Faz 7 ML çekirdekleri: autograd, norm, dropout, conv2d ve pooling
- **Tam autograd kapsamı:** `Variable` + `VarTensor`; aritmetik, broadcast, şekil, 14 aktivasyon ve CNN operatörlerinin geri yayılımı
- **Kayıp fonksiyonları:** fused BCE-with-logits ve softmax cross-entropy dahil 6 hazır loss
- **Custom-op altyapısı:** `custom_unary` / `custom_loss` ile CUDA kernel yazmadan özel fonksiyon tanımı
- Şekil yardımcıları: genel `reshape`, `concat`, `stack`, `one_hot`, genel `broadcast_add/subtract/multiply/divide`, elementwise `divide`
- **Zero-copy view'lar:** `MatrixView` ile kopyasız `transpose_view` / `slice_view` / `reshape_view` / `as_strided_view` + `materialize`
- **In-place operasyonlar:** `add_/sub_/mul_/div_/relu_/sigmoid_/tanh_/clamp_/gelu_/swish_...` + `broadcast_add_/broadcast_multiply_`
- **İndeksleme:** `index_select` / `gather`, `scatter_add`, `embedding` + `embedding_backward`
- **Fused kernel'ler:** tek geçişte `fused_sigmoid_mul`, `fused_relu_add`, `fused_add_mul`, `fused_scale_bias`, `fused_bias_gelu` (+ generic `fused_binary/ternary`)
- **Stream havuzu + async argmax:** 4'lü pool (`pool_stream` / `launch_on_pool`), `__shfl_down_sync` + `float4` tabanlı `argmax_async` / `argmin_async` (pinned host slot + callback)
- **Philox device RNG:** `rng_seed` + `randn_gpu` / `uniform_gpu` / `dropout_gpu` (xorshift yerine sayaç-tabanlı, istatistiksel kalite)
- **Dtype soyutlaması:** `DType` enum, `TypedBuffer<T>`, fp64 referans `sum_f64` / `mean_f64` / `l2_norm_f64`, fp16 `pack_f16` / `unpack_f16`
- **Sparse CSR:** `SparseCSR::from_dense` / `from_coo`, `spmv`, `sparse_matmul`
- Sürümlenmiş dosya biçimi (magic number + versiyon) ve geriye dönük `save/load` uyumluluğu

Bu durum, kütüphaneyi yalnızca işlevsel bir CUDA örneği olmaktan çıkarıp, gerçek performans hedefli bir matris çarpım, GPU hesaplama ve otomatik türev altyapısına dönüştürmektedir.

---

## 🗺️ Yol Haritası

Aşağıdaki maddeler, projenin gelecekte geliştirilebileceği potansiyel alanlardır:

- [ ] Çoklu-GPU desteği (temel altyapı hazır: `device_count` / `set_device` / `DeviceScope` ve cihaz-başına handle; parametre paylaşımı henüz yok)
- [x] Tutarlı hata hiyerarşisi (`MatrixProError` tabanı + alt türler)
- [x] Thread başına CUDA context (çok-threadli kullanım güvenliği)
- [x] `device_only` bellek modu (host kopyası olmadan çalışma)
- [x] FP16 / Tensor Core optimizasyonlarını daha geniş yelpaze ile açmak
- [ ] Python bağlama katmanı (pybind11 ile)
- [x] Daha fazla aktivasyon fonksiyonu (LeakyReLU, GELU, Swish vb.)
- [x] Otomatik türev ve geri yayılım prototiplemesi
- [x] BatchNorm, LayerNorm ve Dropout
- [x] Conv2D ve Max/Average Pooling
- [x] Konvolüsyon ve pooling için geri yayılım (CNN eğitimi)
- [x] ND Tensor (`VarTensor`) üzerinde autograd ve `batch_matmul` gradyanı
- [x] Hazır kayıp fonksiyonları (MSE, MAE, Huber, BCE, Cross-Entropy, KL)
- [x] Genel amaçlı custom-op / custom-loss altyapısı
- [x] Sparse (seyrek) matris desteği (CSR: `from_dense` / `from_coo`, `spmv`, `sparse_matmul`)
- [ ] Linux/CMake çapraz platform derleme desteğinin genişletilmesi
- [x] Akış bazlı overlap ve multi-stream operasyonları (4'lü stream havuzu + `launch_on_pool` + async argmax/argmin)
- [ ] Optimizer katmanı (SGD, Adam) ve parametre güncelleme API'si

---

## 🤝 Katkıda Bulunma

Katkılarınızı memnuniyetle karşılarız! Katkıda bulunmak isterseniz:

1. Depoyu fork'layın
2. Yeni bir özellik dalı (branch) oluşturun (`git checkout -b ozellik/yeni-ozellik`)
3. Değişikliklerinizi commit edin (`git commit -m 'Yeni özellik eklendi'`)
4. Dalınızı push edin (`git push origin ozellik/yeni-ozellik`)
5. Bir Pull Request açın

Hata bildirimi veya özellik önerisi için lütfen [Issues](../../issues) sekmesini kullanın.

---

## 📄 Lisans

Bu proje, depoda yer alan [`LICENSE`](./LICENSE) dosyasında belirtilen lisans koşulları altında dağıtılmaktadır. Kullanmadan önce lisans dosyasını incelemeniz önerilir.

---

## 👨‍💻 Geliştirici Hakkında

Bu kütüphane, **Muhammed Fatih Şahin** tarafından geliştirilmiştir.

> 🎓 **Muhammed Fatih Şahin**, Bursa Teknik Üniversitesi Bilgisayar Mühendisliği bölümü **1. sınıf** öğrencisidir.
> 
> ⚡ Bu proje **yapay zeka desteğiyle** geliştirilmiştir.

Bir öğrencinin CUDA, cuBLAS ve modern C++ ile GPU hızlandırmalı hesaplama dünyasına attığı bu adım, hem öğrenme sürecinin hem de yapay zeka destekli geliştirmenin somut bir örneğidir. Geri bildirimleriniz ve katkılarınız projenin gelişimi için değerlidir.

---

<div align="center">

**⭐ Projeyi beğendiyseniz yıldız vermeyi unutmayın! ⭐**

</div>