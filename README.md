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

- **TF32 Tensor Core matematik modu** etkinleştirildi
- gereksiz ilk `upload()` yükü kaldırıldı
- sıfır matrisler için doğrudan `cudaMemset` kullanıldı
- GPU bellek havuzu ile tekrarlayan alloc/free maliyeti azaltıldı
- pinned host memory ve async transferler eklendi
- reduction akışı global state bağımlılığından kurtarılarak daha güvenli hale getirildi

Bu sayede kütüphane, sadece doğruluk odaklı bir örnek değil, performans odaklı bir CUDA matris altyapısı haline geldi.

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
| **Matris Çarpımı** | cuBLAS ile hızlandırılmış, row-major uyumlu matris çarpımı |
| **Doğrusal Cebir** | `transpose`, `flatten`, `slice`, `trace`, `determinant`, `inverse` |
| **Dış Çarpım** | `outer_product` |

### 📉 GPU Üzerinde İndirgemeler (Reductions)
```
sum · mean · min · max · argmin · argmax · variance · stddev · l1_norm · l2_norm · abs_max
```
Ayrıca eksen bazlı indirgemeler: `row_sum`, `col_sum`

### 🧮 Elementwise Matematiksel Operasyonlar
```
exp · log · sqrt · abs · clamp · pow · add_scalar · sigmoid · tanh · negatif
```

### 📡 Yayınlama (Broadcasting)
- `add_row_vector` — satır vektörünü tüm satırlara yayma
- `add_col_vector` — sütun vektörünü tüm sütunlara yayma
- `multiply_row_vector` / `multiply_col_vector`

### 🎯 Aktivasyon Fonksiyonları
- `relu`
- `softmax` — çok bloklu ve **sayısal olarak kararlı** implementasyon
- `sigmoid`, `tanh`

### 🎲 Matris Oluşturucular
```
zeros · ones · identity · random · uniform · randn · glorot (Xavier init)
```

### 💾 Kalıcı Veri Desteği
- `matrix.save("dosya")` ile matrisi diske yazma
- `Matrix::load("dosya")` ile diskten geri yükleme

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

Proje, sorumlulukların net şekilde ayrıldığı modüler bir dosya yapısına sahiptir:

```text
MatrixFlash_Pro/
│
├── include/matrix_pro/
│   ├── matrix.hpp              # Genel (public) Matrix API tanımı
│   ├── operations.hpp          # GPU operasyon fonksiyon imzaları
│   └── cuda_utils.hpp          # CUDA hata kontrolü ve senkronizasyon yardımcıları
│
├── src/
│   ├── matrix.cu                    # Yaşam döngüsü, kopyalama, save/load işlemleri
│   ├── matrix_factories.cu          # zeros, ones, identity, random/randn/glorot
│   ├── operations_elementwise.cu    # add, subtract, Hadamard çarpımı, skaler işlemler, broadcast
│   ├── operations_matmul.cu         # cuBLAS matris çarpımı + outer_product
│   ├── operations_transforms.cu     # transpose, relu, softmax, flatten, slice
│   ├── operations_statistics.cu     # GPU indirgemeleri (sum/mean/min/max/...)
│   ├── operations_advanced.cu       # determinant ve inverse hesaplamaları
│   └── cuda_utils.cu                # CUDA hata kontrol mekanizması
│
├── tests/                       # Davranış (behavior) testleri
├── benchmarks/
│   └── perf_matrix.cpp          # GPU matris çarpımı benchmark aracı
├── examples/
│   └── basic_gpu_usage.cpp      # Temel kullanım örneği
│
├── CMakeLists.txt
├── LICENSE
└── README.md
```

Bu ayrıştırma sayesinde her `.cu` dosyası tek bir sorumluluğa odaklanır: örneğin matris çarpımı mantığı yalnızca `operations_matmul.cu` içinde, istatistiksel indirgemeler ise yalnızca `operations_statistics.cu` içinde bulunur. Bu yapı, kütüphaneye yeni bir operasyon eklemeyi veya mevcut bir operasyonu hata ayıklamayı önemli ölçüde kolaylaştırır.

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
#include "matrix_pro/matrix.hpp"

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

`exp`, `log`, `sqrt`, `abs`, `clamp`, `pow`, `add_scalar`, `sigmoid`, `tanh`, negatif alma, toplama/çıkarma, Hadamard (elementwise) çarpım

</details>

<details>
<summary><b>🔹 Yayınlama (Broadcasting)</b></summary>

`add_row_vector`, `add_col_vector`, `multiply_row_vector`, `multiply_col_vector`

</details>

<details>
<summary><b>🔹 İstatistik / İndirgemeler</b></summary>

`sum`, `mean`, `min`, `max`, `argmin`, `argmax`, `variance`, `stddev`, `l1_norm`, `l2_norm`, `abs_max`, `row_sum`, `col_sum`

</details>

<details>
<summary><b>🔹 Dönüşümler ve Aktivasyonlar</b></summary>

`transpose`, `flatten`, `slice`, `relu`, `softmax`, `sigmoid`, `tanh`

</details>

<details>
<summary><b>🔹 İleri Düzey Doğrusal Cebir</b></summary>

`trace`, `determinant`, `inverse`, `outer_product`

</details>

<details>
<summary><b>🔹 Kalıcılık (Persistence)</b></summary>

`matrix.save("dosya_yolu")` / `Matrix::load("dosya_yolu")`

</details>

---

## 🧪 Testler

Proje, `tests/` klasörü altında davranış (behavior) testleri içerir. Bu testler, her operasyonun beklenen matematiksel çıktıyı ürettiğini doğrulamak için CMake/CTest altyapısı ile entegre edilmiştir:

```powershell
ctest --test-dir build -C Release --output-on-failure
```

---

## 📈 Performans / Benchmark

`benchmarks/perf_matrix.cpp` dosyası, cuBLAS destekli matris çarpımının GPU performansını karşılaştırmalı biçimde ölçmek için kullanılır. Benchmark, tek bir boyut yerine birden fazla kare matris boyutunu sırayla çalıştırır; her boyut için ortalama, minimum, maksimum süre, GFLOPS ve tahmini bant genişliği değerlerini raporlar.

### Temel kullanım

```powershell
.\build\Release\matrix_pro_benchmark.exe 1024
```

### Çoklu karşılaştırma ve detaylı rapor

```powershell
.\build\Release\matrix_pro_benchmark.exe --sizes 2048,4096,8192,16384 --repeats 3
```

Bu komut aşağıdaki bilgileri üretir:
- her boyut için ortalama / minimum / maksimum çalışma süresi
- GFLOPS oranı
- tahmini bant genişliği (GB/s)
- referans boyutuna göre göreceli hız karşılaştırması
- en yüksek verim elde edilen boyut

### Son doğrulanmış benchmark çıktısı

```text
========================================
 MatrixFlash-Pro GPU Benchmark Report
========================================
Repeats per size: 3 (warm-up run excluded from all measurements)

Size        Avg(ms)     Min(ms)     Max(ms)     GFLOPS      BW(GB/s)        Rel. to base
------------------------------------------------------------------------------------------
2048        18.235      15.241      20.685      942.13      2.76            1.00            x
4096        69.994      69.016      71.024      1963.57     2.88            2.08            x
8192        363.579     341.118     377.545     3024.13     2.21            3.21            x
16384       2735.710    2470.501    2905.017    3215.29     1.18            3.41            x

Interpretation:
- Baseline size: 2048x2048 (reference)
- Relative to baseline > 1.0x indicates faster throughput than the reference size.
- GFLOPS is computed as: 2 * n^3 / elapsed_time_seconds
- Memory bandwidth is estimated from the matrix data movement involved in the multiply workload.

Best throughput observed: 16384x16384 with 3215.29 GFLOPS
```

Bu sonuçlar, kütüphanenin son optimize edilmiş versiyonunda matris çarpımının hem büyüklük ölçeğinde yüksek verim verdiğini hem de büyük matrislerde daha iyi göreceli performans sağladığını göstermektedir. Özellikle 16384×16384 matris boyutunda görülen 3215.29 GFLOPS, proje için kritik başarı ölçütüdür.

---

## ✅ Son Durum

MatrixFlash-Pro şu anda aşağıdaki temel optimizasyonları içermektedir:

- **TF32 Tensor Core matematik modu**
- gereksiz host→device upload kaldırıldı
- sıfır matrisler için doğrudan `cudaMemset`
- bellek havuzu ile alloc/free maliyeti azaltıldı
- pinned host memory ve async transferler
- stream-safe reduction yapısı
- cuSOLVER tabanlı ileri düzey doğrusal cebir destekleri

Bu durum, kütüphaneyi yalnızca işlevsel bir CUDA örneği olmaktan çıkarıp, gerçek performans hedefli bir matris çarpım ve GPU hesaplama altyapısına dönüştürmektedir.

---

## 🗺️ Yol Haritası

Aşağıdaki maddeler, projenin gelecekte geliştirilebileceği potansiyel alanlardır:

- [ ] Çoklu-GPU desteği
- [ ] FP16 / Tensor Core optimizasyonlarını daha geniş yelpaze ile açmak
- [ ] Python bağlama katmanı (pybind11 ile)
- [ ] Daha fazla aktivasyon fonksiyonu (LeakyReLU, GELU, Swish vb.)
- [ ] Sparse (seyrek) matris desteği
- [ ] Linux/CMake çapraz platform derleme desteğinin genişletilmesi
- [ ] Akış bazlı overlap ve multi-stream operasyonları daha da yaygınlaştırmak

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