# MatrixFlash-Pro: optimizasyon ve rakip benchmark raporu

**Tarih:** 1 Ekim 2026. **Ölçülen donanım:** RTX 3070 Laptop, Windows.

Bu çalışma, mevcut matris API'sini ve host/device aynası sözleşmesini koruyarak
GEMM yürütme maliyetlerini azalttı; tekrar kullanılabilir GEMM ve LU planları,
açık hassasiyet politikaları ve daha kararlı ileri doğrusal cebir ekledi.
Sonuçlar yalnızca aşağıdaki donanım, boyutlar ve ölçüm kapsamı için geçerlidir.
Tek bir laptopta yapılan ölçüm, “dünyanın en hızlı/en güçlü kütüphanesi” sonucunu
kanıtlamaz. İyileşmeyen ve gerileyen durumlar tablolarda özellikle bırakılmıştır.

Ölçülen en belirgin kazanım, `1×4096 · 4096×1024` default çarpımında
**185.824 → 46.656 µs (3.98×)**. `31×19 · 19×47` default
yolunda **1.31×**,
`127×63 · 63×257` yolunda **1.89×**.
`1024×512 · 512×64` için açık TF32 tuned planı eski default'a göre
**2.41×** hızlıdır. `1024²`/`2048²` default ölçümleri baseline'dan
sırasıyla **%17.7** /
**%10.7** daha yavaş çıktı.
Küçük şekillerdeki dağılımlar, kazanımlar ve gerilemeler de tablolarda bırakıldı;
kontrollü saat/scheduler ölçümü olmadan küçük farklara nedensel hız iddiası yapılmıyor.

## 1. Teslim edilen değişiklikler

### Derleme ve mevcut davranışların korunması

- CUDA 13'te kaldırılan device-property alanları ve değişen prefetch imzası
  güncellendi. Yanlış include yolları, eksik bildirimler ve eski API kullanımları
  onarıldı. Kullanıcının verdiği CUDA mimarisi artık ezilmiyor; varsayılan `native`.
- Ayrı `build/perf-release` dizini ve Windows/MSVC ortamını kuran
  `tools/build_local.py` eklendi. Tam Release derlemesi: kütüphane, 26 test
  programı, mevcut benchmarklar ve örnekler başarılı.
- Serialization cihaz adresleri ve host aynası işaretleri düzeltildi.
  Autograd'da no-grad/detach, toplam türevi ve checkpoint upstream gradyanı
  onarıldı. Pipeline stream seçimi ve exception sonrası geri dönüş düzeltildi.
- Önceden çalışma alanında silinmiş `doc/`, `.idea/`, `.vscode/` dosyaları
  bu çalışmanın commitlerine dahil edilmedi.
- NVIDIA bellek denetimiyle bulunan eski pool kapanış double-free hatası
  onarıldı: cached/live blokların ortak ownership map'i bir kez temizlenir,
  free işlemi doğru device üzerinde yapılır. Power-of-two allocation sınıfı
  taşması sonsuz döngüye girmeden reddedilir.

### Daha düşük GEMM yürütme maliyeti

- Önceden her çağrıda oluşturulan cuBLASLt descriptor, layout ve heuristic
  sonuçları tekrar kullanılıyor. Cache üst sınırı host thread başına 128 plan.
- Mutable handle/descriptor/workspace artık host thread, cihaz ve stream'e
  göre ayrılıyor. Bağımsız stream'ler aynı scratch belleği kullanmıyor.
- Çalışma bağlamı başına paylaşılan 64 MiB workspace var; plan başına 64 MiB
  ayrılmıyor. `workspace_limit_bytes`, Lt algoritmasının kullanabileceği limiti
  belirler; bağlamın rezervasyon boyutunu küçültmez. Mevcut çekirdek handle'larının
  workspace'i ayrıca bulunabilir; bu değişiklik belleksiz bir hızlandırma değildir.
- Default ve fused GEMM aynı plan motorunu kullanıyor. Sıcak raw dispatch
  tekrar eden stream sorgularını atlıyor. Küçük matrislerin mikro kernel yolu
  ve matrix × vector yolu korunuyor; uzun vector × matrix cuBLASLt'ye yönleniyor.
- Özel vector × matrix kernel'ında K boyutu 8 warp'a bölünüp ortak bellekte
  birleştiriliyor. Uzun K'de ölçülen daha hızlı vendor yolu tercih ediliyor.
- Uzun vector default yolu vendor backend'e taşınsa da strict FP32 input
  hassasiyeti korunur; vektör hızlanması TF32'ye düşürülerek elde edilmez.
- K=0, output aliasing, int32 backend sınırı ve allocation taşması açıkça
  ele alınıyor. `multiply_into` ve eski çarpım çağrıları aynı imzalarla devam ediyor.

### Yeni ileri API'ler

| API | Sağladığı davranış |
|---|---|
| `GemmPlan` | Önceden hazırlanmış, tahsissiz sıcak GEMM yürütmesi |
| `GemmOptions` | A/B transpose, FP32/TF32, workspace limiti, fused bias/ReLU/GELU |
| `execute(A,B,C,alpha,beta,bias)` | `C = epilogue(alpha op(A) op(B) + beta C)` |
| `tune(...)` | En fazla 16 Lt heuristic adayını ölçer; scratch ile C'yi korur |
| `gemm` / `gemm_into` | Sınırlı plan cache'i kullanan kolaylık API'leri |
| `LUFactorization` | A'yı bir kez faktörler; pivotlar ve RHS scratch'i tekrar kullanır |
| `solve_into` | Önceden ayrılmış çıktıda çoklu RHS / transpose çözümü |
| `slogdet` | Determinant işareti + double log-mutlak-değer; FP32 ürün taşmasını önler |
| `matrix_exp` | Varsayılan [13/13] Padé + scaling/squaring; iç çarpımlar strict FP32 |

`matrix_exp(A, order)` alternatif 1–64 dereceler için ölçeklenmiş Taylor kullanır.
Bu hâlâ FP32 matris kütüphanesidir: double log birikimi, tüm işlemlerin FP64 olduğu
anlamına gelmez. `matrix_power` negatif INT_MIN taşmasını önler; `batch_det`
negatif determinantın işaretini artık kaybetmez.

### Seyrek matris doğruluğu ve karmaşıklığı

- Sırasız COO triplet'ları sıralanır, aynı koordinattaki değerler toplanır.
- Dikdörtgen CSC↔CSR dönüşümü ve CSC transpose onarıldı. CSC→CSR'de dense
  ara matris kaldırıldı; host üzerinde O(nnz) veri çıkarma ve sıralama kullanılır.
- `sparse_add` gerçekten `alpha A + beta B` hesaplar; eski placeholder kaldırıldı.
- Host dönüşümleri nonblocking compute stream'i bekler; eski stream yarışı giderildi.
- COO kernel'larının grid-stride döngüsü 200.000'inci kayıt dahil tamamını işler.
- Boş CSR'nin dense, SpMV ve sparse×dense sonuçları cihaz üzerinde sıfırlanır.

Bu dönüşümler tamamen GPU'da çalışan yeni bir sparse motor olarak sunulmuyor.
Bazıları host sıralaması/senkronizasyonu içerir; CSC dense çarpım fallback'i gibi
eski yolların ayrıca optimize edilmesi gerekir. Sparse hız kazanımları ölçülmedi.

## 2. Doğrulama

**CTest: 26/26 test programı başarılı** (17 unit, 9 integration).
Orijinal derlenebilir baseline'da 24/24 başarılıydı; iki yeni program GEMM planı
ve factorization özelliklerini sınar. Sparse dönüşüm testi ayrıca arka arkaya
üç çalıştırmada geçti. Yeni planlı GEMM/LU örneği çalıştırıldı.
NVIDIA Compute Sanitizer memcheck, GEMM planı (50.462 check), factorization
(42 check) ve sparse format (62 check) programlarında **0 hata** bildirdi.
Bu memcheck sonucu tüm olası race'lerin veya leak'lerin yokluğunu kanıtlamaz.

Yeni/ genişletilen kontroller:

- 4 transpose kombinasyonu × 4 epilogue, alpha/beta ve bağımsız CPU double dot.
- K=0, alias/shape/int32 hataları, taşınmış plan ve bayat host aynası.
- Tuning sırasında beta girdisinin değişmemesi; CUDA Graph replay ile değişen
  input buffer'larının okunması; aynı anda iki host thread'de doğru GEMM sonucu.
- İki Pipeline stream'inde GEMM, exception sonrası stream'in geri yüklenmesi,
  yarım kalmış graph capture'ının reset ile temizlenmesi.
- Pivotlu LU, çoklu RHS, transpose çözümü, singular/boş/non-square durumlar,
  capture sırasında factor oluşturmanın reddi; residual matematik kontrolleri.
- FP32 aralığını aşan determinant için sonlu log değeri, pivot işareti,
  `diag(8,-8)` üssü ve 12 radyan dönme matrisinin sin/cos referansı.
- Yinelenen/sırasız COO, dikdörtgen CSC roundtrip/transpose, alpha/beta sparse
  toplamı, geçersiz indeks, büyük COO kuyruğu ve boş CSR çıktıları.

Benchmark çıktıları da zamanlamadan önce ve sonra en fazla 97 deterministik
konumda bağımsız double dot-product ile kontrol edildi. Büyük benchmark
matrislerinin her elemanı karşılaştırılmadı; küçük plan testleri elemanları kapsar.
Tolerans: `1e-5 + sum(abs(a_i*b_i)) * (2e-6 FP32 veya 0.002 TF32)`.
NaN/sonsuz çıktı hata sayılır. Bu sınır, her probleme uygun hata garantisi değildir.

Son GPU turlarında örneklenen en büyük mutlak hata: strict FP32 **1.1248941e-05**, TF32 **0.0033099825**.
Bu değerler bu girdilere/örneklenen konumlara aittir; genel hata sınırı değildir.

## 3. Donanım, yazılım ve ölçüm kapsamı

| Alan | Değer |
|---|---|
| GPU | NVIDIA GeForce RTX 3070 Laptop GPU, 8 GiB, SM 86 |
| CPU | AMD Ryzen 7 5800H with Radeon Graphics |
| Platform | Windows 11, WDDM; laptop güç/termal koşulları |
| NVIDIA sürücüsü | 610.88 |
| CUDA compiler toolkit | 13.4, nvcc 13.4.59 |
| CUDA runtime/driver API | Ham JSON'daki `cuda_runtime` / `cuda_driver` değerleri |
| Derleme | C++17, Release, MSVC 14.44 / VS 2022 Preview, native SM86, cuDNN kapalı |
| CPU rakibi | NumPy 2.3.5, OpenBLAS 0.3.30; runtime thread sayısı sabitlenmedi |
| Girdiler | FP32 uniform [-0.5,0.5], seed 20261001 |
| GPU tekrarları | 10 warmup, 30 ölçüm × 32 GEMM; son kod için iki ayrı tur |

Baseline, kaynak kodun ilk hâlindeki GEMM dispatch/kernel'larını kullanır;
yalnızca derlemeyi ve mevcut testleri çalıştırmayı sağlayan uyumluluk düzeltmeleri
uygulanmıştır. Baseline'da 70, her son turda 182 engine/policy/shape satırı vardır.
Tablolar son iki turun 60 örneğinin ortak medyanıdır; tek turun en iyi değeri
seçilmemiştir. Baseline tek tur/30 örnektir.

GPU girdileri/çıktıları önceden ayrılmıştır; upload/download, allocation,
descriptor/heuristic oluşturma ve tuning süresi zamanlamanın dışındadır.
cuBLAS ve cuBLASLt aynı veri, precision politikası, compute stream ve 64 MiB
workspace altında ölçülür. Ham Lt descriptor ve heuristic'leri de tekrar kullanır.
Dolayısıyla rakip “soğuk” başlatılarak kütüphane lehine sonuç üretilmemiştir.

`device_samples_ms`, CUDA event'leri arasındaki **stream elapsed** süresidir;
host'un komut gönderirken bıraktığı boşlukları da içerir, salt kernel süresi değildir.
`wall_samples_ms` batch sonu beklemeyi de içerir. CUDA Graph tablosu ayrı kapsamdadır:
bir graph'a 32 GEMM kaydedilir, replay süresi 32'ye bölünür; bağımsız 32 graph
launch'u olarak ölçülmez. Tuning üç kısa denemenin medyanıyla seçim yapar;
aday sayısı/submit maliyeti ve gürültü nedeniyle her şekli hızlandırması garanti değildir.

GPU saatleri/güç limiti kilitlenmedi; ekran ve diğer Windows aktiviteleri
izole edilmedi. Derleme, CPU rakip ölçümü ve GPU benchmarklar eşzamanlı
çalıştırılmadı. Küçük farklara istatistiksel üstünlük atfedilmemelidir.

## 4. Eski ve yeni varsayılan çarpım

Şekil `M × N × K`: A, M×K; B, K×N. Süreler mikro saniye (µs).
Hız katsayısı = eski medyan / yeni medyan; **1'den küçük değer gerilemedir**.
Default API'nin eski şekle bağlı politikası korunur: küçük/GEMV yolları ile
TF32 vendor yolları karışıktır. Tuned sütunu açık TF32 planını gösterir;
default'un küçük şekillerdeki FP32 kernel'ıyla aynı hassasiyet sayılmaz.

| M × N × K | Eski default | Yeni default | Eski/yeni | TF32 tuned | Yeni default P10–P90 |
|---|---:|---:|---:|---:|---:|
| 16 × 16 × 16 | 10.192 | 9.488 | 1.07× | 14.224 | 8.438–11.399 |
| 32 × 32 × 32 | 14.560 | 11.999 | 1.21× | 20.352 | 10.061–15.306 |
| 64 × 64 × 64 | 15.584 | 15.392 | 1.01× | 15.248 | 12.573–17.766 |
| 128 × 128 × 128 | 16.112 | 18.208 | 0.88× | 15.920 | 12.832–25.648 |
| 256 × 256 × 256 | 14.112 | 14.336 | 0.98× | 20.912 | 12.749–21.008 |
| 512 × 512 × 512 | 34.832 | 31.344 | 1.11× | 27.792 | 30.365–48.016 |
| 1024 × 1024 × 1024 | 159.376 | 187.584 | 0.85× | 176.240 | 164.381–203.994 |
| 2048 × 2048 × 2048 | 1164.528 | 1288.800 | 0.90× | 1307.232 | 1257.741–1375.478 |
| 31 × 47 × 19 | 18.624 | 14.240 | 1.31× | 13.776 | 9.219–33.917 |
| 127 × 257 × 63 | 26.240 | 13.856 | 1.89× | 15.088 | 12.128–18.458 |
| 64 × 1024 × 512 | 36.752 | 23.856 | 1.54× | 18.672 | 21.136–35.344 |
| 1024 × 64 × 512 | 39.264 | 24.720 | 1.59× | 16.304 | 21.040–39.693 |
| 1 × 1024 × 4096 | 185.824 | 46.656 | 3.98× | 46.736 | 44.054–60.419 |
| 1024 × 1 × 4096 | 42.368 | 42.976 | 0.99× | 47.904 | 42.141–59.574 |

P10–P90, 60 örnekteki dağılım aralığıdır; güven aralığı değildir. Baseline ile
son ölçümler farklı zamanlarda alındığı için güç/saat ve scheduler farkları
özellikle büyük kare matrislerde etkili olabilir. Aşağıdaki aynı-tur vendor
tablosu wrapper/algoritma davranışını daha doğrudan karşılaştırır.

## 5. Rakip GPU kütüphaneleri: aynı hassasiyet

FP32: `CUBLAS_COMPUTE_32F_PEDANTIC`; azaltılmış input mantissa dönüşümü kapalı.
TF32: `CUBLAS_COMPUTE_32F_FAST_TF32`; input doğruluğundan açıkça ödün veren
Tensor Core politikası. FP32 depolama ve birikim devam eder. TF32 sonucunu strict
FP32 rakibe karşı aynı hassasiyetmiş gibi sıralamak doğru değildir.

Tuned plan en fazla 16 heuristic adayı ölçer; raw Lt ilk heuristic'i kullanır.
Olası tuned üstünlüğü, NVIDIA backend'inin algoritmalarını önceden seçmekten gelir.
Tuning maliyeti ve ilk plan oluşturma burada amortize edilmiştir.

### FP32 — önceden ayrılmış normal gönderim, µs

| M × N × K | Plan | Tuned plan | Ham cuBLAS | Ham cuBLASLt |
|---|---:|---:|---:|---:|
| 16 × 16 × 16 | 12.976 | 12.864 | 13.840 | 12.352 |
| 32 × 32 × 32 | 12.096 | 12.384 | 14.176 | 12.096 |
| 64 × 64 × 64 | 14.848 | 15.344 | 15.328 | 13.200 |
| 128 × 128 × 128 | 12.928 | 14.784 | 27.760 | 12.702 |
| 256 × 256 × 256 | 14.240 | 14.048 | 15.872 | 15.056 |
| 512 × 512 × 512 | 43.584 | 45.920 | 48.032 | 48.384 |
| 1024 × 1024 × 1024 | 364.864 | 276.704 | 343.440 | 347.376 |
| 2048 × 2048 × 2048 | 2339.488 | 2040.656 | 2388.608 | 2364.528 |
| 31 × 47 × 19 | 18.112 | 15.344 | 16.960 | 13.120 |
| 127 × 257 × 63 | 13.472 | 13.600 | 14.288 | 12.880 |
| 64 × 1024 × 512 | 24.992 | 15.968 | 34.512 | 28.304 |
| 1024 × 64 × 512 | 28.063 | 17.488 | 28.512 | 26.400 |
| 1 × 1024 × 4096 | 45.392 | 48.320 | 45.488 | 45.296 |
| 1024 × 1 × 4096 | 48.096 | 47.552 | 45.520 | 47.792 |

### TF32 — önceden ayrılmış normal gönderim, µs

| M × N × K | Plan | Tuned plan | Ham cuBLAS | Ham cuBLASLt |
|---|---:|---:|---:|---:|
| 16 × 16 × 16 | 12.704 | 14.224 | 14.272 | 11.840 |
| 32 × 32 × 32 | 14.768 | 20.352 | 14.880 | 12.640 |
| 64 × 64 × 64 | 13.536 | 15.248 | 20.240 | 14.112 |
| 128 × 128 × 128 | 13.824 | 15.920 | 20.480 | 12.814 |
| 256 × 256 × 256 | 13.824 | 20.912 | 18.624 | 13.584 |
| 512 × 512 × 512 | 36.128 | 27.792 | 35.200 | 34.848 |
| 1024 × 1024 × 1024 | 186.256 | 176.240 | 169.168 | 171.472 |
| 2048 × 2048 × 2048 | 1320.544 | 1307.232 | 1324.896 | 1304.064 |
| 31 × 47 × 19 | 18.448 | 13.776 | 16.349 | 13.807 |
| 127 × 257 × 63 | 14.062 | 15.088 | 17.680 | 13.888 |
| 64 × 1024 × 512 | 26.016 | 18.672 | 27.600 | 23.872 |
| 1024 × 64 × 512 | 23.776 | 16.304 | 26.159 | 32.768 |
| 1 × 1024 × 4096 | 46.640 | 46.736 | 45.216 | 45.504 |
| 1024 × 1 × 4096 | 47.216 | 47.904 | 48.048 | 48.160 |

## 6. CUDA Graph: aynı 32-GEMM batch kapsamı

Graph, küçük/orta işlemlerde host submission maliyetini amortize eder.
Her çağrının output'unu host'ta okumayı gerektiren bir uygulama aynı kazancı
alamaz. Aynı kapsamlı ham Lt graph da ölçülmüştür.

| M × N × K | FP32 plan graph | FP32 ham Lt graph | TF32 plan graph | TF32 ham Lt graph |
|---|---:|---:|---:|---:|
| 16 × 16 × 16 | 3.040 | 2.144 | 6.864 | 3.424 |
| 32 × 32 × 32 | 4.032 | 4.224 | 3.424 | 3.616 |
| 64 × 64 × 64 | 4.880 | 4.896 | 4.288 | 3.936 |
| 128 × 128 × 128 | 4.928 | 4.832 | 4.896 | 4.928 |
| 256 × 256 × 256 | 8.464 | 8.144 | 8.880 | 7.248 |
| 512 × 512 × 512 | 45.152 | 47.072 | 24.720 | 31.824 |
| 1024 × 1024 × 1024 | 269.008 | 343.840 | 166.064 | 166.624 |
| 2048 × 2048 × 2048 | 2040.272 | 2386.160 | 1317.504 | 1312.864 |
| 31 × 47 × 19 | 3.648 | 3.616 | 10.544 | 3.648 |
| 127 × 257 × 63 | 4.160 | 5.584 | 4.479 | 4.128 |
| 64 × 1024 × 512 | 13.360 | 13.230 | 14.144 | 10.928 |
| 1024 × 64 × 512 | 14.384 | 16.304 | 13.520 | 10.912 |
| 1 × 1024 × 4096 | 42.144 | 42.480 | 44.144 | 42.432 |
| 1024 × 1 × 4096 | 42.432 | 42.320 | 42.320 | 42.592 |

## 7. NumPy / OpenBLAS ve isteğe bağlı framework'ler

NumPy CPU'da, MatrixFlash GPU'da önceden ayrılmış buffer'larla çalışır.
Bu tablo cihazda hazır matrislerin uygulama tercihi için bağlam verir;
GPU-kernel/CPU oranını “rakibin kütüphane overhead'ini yenme” olarak yorumlamaz.
İki tarafta da FP32 depolama vardır; GPU sütunu strict FP32 planıdır.
Girdilerin dağılımı ve seed'i aynı olsa da C++ mt19937, NumPy PCG64 kullandığı
için değerler bire bir aynı değildir. Veri transfer maliyeti tabloda yoktur.
CPU BLAS thread sayısı sabitlenmedi; OpenBLAS build yapılandırmasındaki
MAX_THREADS=24, çalışmada 24 thread kullanıldığına dair kanıt değildir.

| M × N × K | GPU strict FP32 plan wall, ms | NumPy CPU wall, ms |
|---|---:|---:|
| 16 × 16 × 16 | 0.013534 | 0.002200 |
| 32 × 32 × 32 | 0.012591 | 0.003550 |
| 64 × 64 × 64 | 0.015983 | 0.009900 |
| 128 × 128 × 128 | 0.013428 | 0.086750 |
| 256 × 256 × 256 | 0.014941 | 0.261400 |
| 512 × 512 × 512 | 0.044550 | 1.093900 |
| 1024 × 1024 × 1024 | 0.367012 | 4.910200 |
| 2048 × 2048 × 2048 | 2.341223 | 41.155150 |
| 31 × 47 × 19 | 0.019300 | 0.003700 |
| 127 × 257 × 63 | 0.014238 | 0.098400 |
| 64 × 1024 × 512 | 0.025606 | 0.372500 |
| 1024 × 64 × 512 | 0.028745 | 0.298850 |
| 1 × 1024 × 4096 | 0.047116 | 0.407000 |
| 1024 × 1 × 4096 | 0.049000 | 0.359300 |

PyTorch ve CuPy, ölçümün yapıldığı Python 3.12 ortamında kurulu değildi;
**ölçülmedi**. `tools/bench_rivals_fair.py` kurulu olduklarında PyTorch için
açık FP32/TF32 CUDA-event ölçümleri, CuPy için ayrıca etiketlenmiş framework
default ölçümü yapabilir. Bu rapor onlar hakkında performans iddiası içermez.

## 8. Kullanım sözleşmesi ve sınırlar

- `GemmPlan` ve `LUFactorization` oluşturulduğu host thread, device ve compute
  stream'e bağlıdır. Başka bağlamda execute/solve hata verir. Aynı planı
  eşzamanlı stream'lerde paylaşmayın; ayrı planlar oluşturun.
- Buffer'lar doğru cihazda, contiguous ve GEMM output'u A/B/bias'tan ayrı
  olmalıdır. `beta != 0` için C'nin **device içeriği** önceden geçerli olmalıdır.
- Planı oluşturun, tuning gerekiyorsa yapın, buffer'ları hazırlayıp sıcak execute
  çağırın; ardından capture edin. Plan ve tüm input/output/bias buffer'ları
  graph'ın ömrü boyunca yaşamalıdır. Capture içinde oluşturma/tuning yapılmaz.
- Uzun ömürlü graph'lar için sahipliği açık `GemmPlan` kullanın. Convenience
  cache'inin churn/eviction davranışına graph kaynaklarının ömrünü emanet etmeyin.
- Graph replay sırasında her GPU yazma çağrısı host'ta tekrar çalışmadığı için,
  host aynasına daha önce eriştiyseniz tekrar replay sonrasında
  `mark_host_stale()` + `download()` kullanın. Host okumaları capture dışında yapılır.
- `LUFactorization` constructor'ı singularity ve signed logdet için senkronize
  eder; `solve_into` preallocated RHS kapasitesinde tahsis yapmaz. `solve` çıktı
  oluşturur. RHS ile output aynı buffer olabilir; önce internal scratch'e kopyalanır.
- Çoklu GPU, Hopper/Blackwell, Linux ve cuDNN yolları bu çalışmada test edilmedi.
  Mevcut tüm ileri operasyonlar için evrensel stabilite/performance garantisi yoktur.
- Fused epilogue, LU reuse, sparse dönüşüm ve matrix-exp için bu çalışmada ayrı
  hız tabloları üretilmedi; bu alanlarda ölçülmemiş hız katsayısı iddiası yapılmaz.

## 9. Yeniden üretim

VS C++ araçları, CMake ve CUDA PATH üzerinde olmalı. Komutları repo kökünde,
Python 3.12 ve mevcut bir NVIDIA GPU ile çalıştırın. Python yolu makinenize göre
değişebilir. Script MSVC ortamını kurar; cuDNN bağımlılığı bu profile kapalıdır.

```powershell
python tools/build_local.py --reconfigure --jobs 4
ctest --test-dir build/perf-release -C Release --output-on-failure -j 1
ctest --test-dir build/perf-release -C Release -R sparse_formats --repeat until-fail:3 --output-on-failure
.\build\perf-release\examples\matrix_pro_demo_planned_gemm.exe
.\build\perf-release\benchmarks\matrix_pro_bench_gemm_fair.exe --output benchmarks/results/run1.json --tag reproduced_run1 --warmup 10 --repeats 30 --batch 32
.\build\perf-release\benchmarks\matrix_pro_bench_gemm_fair.exe --output benchmarks/results/run2.json --tag reproduced_run2 --warmup 10 --repeats 30 --batch 32
python tools/bench_rivals_fair.py --output benchmarks/results/rivals_fair.json --warmup 10 --repeats 30 --batch 32
```

GPU ve CPU benchmarklarını/derlemeyi paralel başlatmayın. Baseline verisi
uyumluluk commit'i `2210ffc` üzerinden kaydedildi; son kod baseline script'inin
aynı default/vendor vakalarını genişletilmiş plan/graph ölçümleriyle sürdürür.

Ham kanıtlar: [`benchmarks/reports/2026-10-01/`](benchmarks/reports/2026-10-01/).
JSON dosyalarında her örnek, medyan, standart sapma, GFLOPS ve örneklenmiş
mutlak hata bulunur. Test ve örnek çalışma logları aynı dizindedir.

## 10. Sonraki ölçülebilir hedefler

1. En az Ampere/Hopper/Blackwell cihazlarda saat/power kontrolüyle tekrar;
   farklı CUDA sürümleri ve OS profilleri, çoklu GPU doğruluk testleri.
2. Tutulan-out workload üzerinde autotuning kararlarını sınama; cache'e
   kalıcı tuning profili, memory bütçesi ve uygulama şekillerine göre dispatch.
3. End-to-end training/inference, transfer dahil latency, fused epilogue ve
   aynı A üzerinde yüzlerce RHS için LU amortizasyonu benchmarkları.
4. Sparse dönüşümleri GPU radix sort/reduce ile taşıma; SpMV/SpMM için gerçek
   sparsity dağılımları ve vendor cuSPARSE karşılaştırması.
5. FP64/karma hassasiyet, iterative refinement ve batched reusable factorization
   için ayrı tasarım, matematik testleri ve performans kanıtı.

Bu hedefler teslim edilmiş özellikler olarak sayılmamıştır.

Teknik kaynak: NVIDIA'nın [cuBLAS/cuBLASLt dokümantasyonu](https://docs.nvidia.com/cuda/cublas/index.html)
(compute precision, epilogue ve heuristic API'leri).

## 11. Sıralı commitler

| Commit | Tamamlanan adım |
|---|---|
| `c25c412` | Ölçülebilir optimizasyon/uyumluluk planı |
| `2210ffc` | CUDA 13 derleme uyumluluğu, ilk 24-test baseline ve ham ölçüm |
| `c83bcd0` | Ownership map ile pool double-free / allocation taşması düzeltmesi |
| `441e095` | Reusable GEMM/LU, açık hassasiyet, fused execution, ileri sayısal işlemler |
| `942aa9b` | Sparse format ve toplama doğruluk düzeltmeleri |
| `124d262` | Uzun vector dispatch'te eski FP32 doğruluğunun açıkça korunması |
| Son rapor commit'i | Yeni benchmarklar, iki son ölçüm turu, memcheck/test kanıtı ve README |

GitHub'ın mevcut `160565a` dokümantasyon geçmişi korunarak `main` ile birleştirilir;
çalışma alanındaki önceden silinmiş dokümantasyon dosyaları bu merge sırasında
diskte yeniden oluşturulmaz. Çalışmanın değişiklikleri source ve test commitleri
üzerinden tek tek incelenebilir.
