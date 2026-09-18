'use client'

import Link from 'next/link'
import { ArrowRight, CheckCircle2, Cpu, ShieldAlert, Zap, Terminal, Layers } from 'lucide-react'
import { CodeBlock } from '@/components/code-block'
import {
  Callout,
  DocSection,
  InlineCode,
  PageHeader,
  SubHeading,
} from '@/components/docs/doc-ui'
import { useLanguage } from '@/lib/language-context'
import { site } from '@/lib/site'

const cloneCode = `# Depoyu klonlayın / Clone repository
git clone https://github.com/muhamedsahin/MatrixFlash_Pro.git
cd MatrixFlash_Pro`

const cmakeBuildCode = `# 1. CMake ile yapılandırın (Release modu önerilir)
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release

# 2. Projeyi ve hedefleri derleyin
cmake --build build --config Release`

const runTestsCode = `# Testleri çalıştırın (Windows PowerShell)
& .\\build\\Release\\matrix_pro_tests.exe

# Benchmark çalıştırın (2048x2048)
& .\\build\\Release\\matrix_pro_benchmark.exe 2048

# Örnek programları çalıştırın
& .\\build\\Release\\matrix_pro_basic.exe
& .\\build\\Release\\matrix_pro_demo_high_perf.exe`

const lifecycleCode = `#include "matrix_pro/matrix_pro.hpp"
#include <iostream>

int main() {
    using namespace matrix_pro;

    try {
        // -------------------------------------------------------------
        // ADIM 1: Bellek Modu ile Matris Oluşturma (Allocation)
        // MemoryMode::device_only ile host kopyası oluşturulmaz (0 PCIe yükü)
        // -------------------------------------------------------------
        Matrix a(1024, 1024, MemoryMode::device_only);
        Matrix b(1024, 1024, MemoryMode::device_only);

        // -------------------------------------------------------------
        // ADIM 2: Cihaz İçi Veri Üretimi / Doldurma (Direct On-Device)
        // Host üzerinden upload() yerine doğrudan GPU kernel ile üretim
        // -------------------------------------------------------------
        a.fill(1.5f);
        b = Matrix::randn_gpu(1024, 1024);

        // -------------------------------------------------------------
        // ADIM 3: Yürütme & Zincirleme (cuBLAS GEMM + Fused Chains)
        // -------------------------------------------------------------
        Matrix prod = a * b; // cuBLAS SGEMM + TF32 Tensor Core
        
        // Fused kernel: bias ekleme ve GeLU tek kernel'da çalışır
        Matrix bias = Matrix::zeros(1024, 1024);
        Matrix activated = fused_bias_gelu(prod, bias);

        // Bellek dostu in-place (yerinde) işlem
        relu_(activated);

        // -------------------------------------------------------------
        // ADIM 4: Asenkron İşlemler (Stream Pool & Async Reductions)
        // -------------------------------------------------------------
        // Argmax hesaplamasını CPU'yu bloklamadan pinned memory'e yazdırın
        std::size_t* max_idx = static_cast<std::size_t*>(pin_host(sizeof(std::size_t)));
        argmax_async(activated, max_idx, [](std::size_t best) {
            std::cout << "Async Callback: En buyuk indeks -> " << best << "\\n";
        });

        // -------------------------------------------------------------
        // ADIM 5: Senkronizasyon & Host İndirme (Fail-Fast Contract)
        // DİKKAT: GPU yazımı sonrası download() çağrılmazsa at() fırlatır!
        // -------------------------------------------------------------
        activated.download(); // Host mirror'ı GPU'dan günceller
        std::cout << "Sonuc hucre (0,0): " << activated.at(0, 0) << "\\n";

        // Belleği serbest bırakın
        unpin_host(max_idx);
    }
    catch (const MatrixProError& e) {
        // Tüm kütüphane istisnaları tek tabandan yakalanır
        std::cerr << "MatrixFlash-Pro Hatasi: " << e.what() << "\\n";
        return 1;
    }

    return 0;
}`

const autogradLifecycleCode = `#include "matrix_pro/matrix_pro.hpp"
#include <iostream>

int main() {
    using namespace matrix_pro;

    // 1. Değişkenleri oluşturun (requires_grad = true)
    Variable x(Matrix::randn(32, 64), /*requires_grad=*/true);
    Variable w(Matrix::randn(64, 10), /*requires_grad=*/true);
    Variable b(Matrix::zeros(1, 10),  /*requires_grad=*/true);
    Variable target(Matrix::ones(32, 10), /*requires_grad=*/false);

    // 2. İleri Yayılım (Forward Pass - Hesaplama grafiği otomatik kaydedilir)
    Variable logits = x.matmul(w).broadcast_add(b);
    Variable loss = mse_loss(logits, target);

    std::cout << "Egitim kaybi (Loss): " << loss.value().mean() << "\\n";

    // 3. Geri Yayılım (Backward Pass)
    loss.backward();

    // 4. Gradyanları okuma & parametre güncelleme
    const Matrix& grad_w = w.grad();
    const Matrix& grad_b = b.grad();
    std::cout << "W gradyan Frobenius normu: " << grad_w.frobenius_norm() << "\\n";

    // 5. Yeni adım için gradyanları sıfırlama
    w.zero_grad();
    b.zero_grad();

    return 0;
}`

export default function DocsHomePage() {
  const { lang, t } = useLanguage()

  return (
    <article className="space-y-4">
      <PageHeader
        eyebrow={t('BAŞLANGIÇ & MİMARİ', 'GETTING STARTED & ARCHITECTURE')}
        title={t('Genel Bakış ve Kurulum', 'Overview & Installation Guide')}
        description={t(
          'MatrixFlash-Pro kütüphanesinin kurulumu, derleme adımları, donanım gereksinimleri, fonksiyon çağrı sırası ve bellek modeli hakkında kapsamlı rehber.',
          'Comprehensive guide to installing, configuring, compiling, calling sequence, and memory lifecycle of the MatrixFlash-Pro library.',
        )}
      />

      {/* Bölüm 1: Proje Hakkında */}
      <DocSection title={t('MatrixFlash-Pro Nedir?', 'What is MatrixFlash-Pro?')}>
        <p>
          {lang === 'tr' ? (
            <>
              <strong className="text-foreground">MatrixFlash-Pro</strong>; NVIDIA GPU’lar üzerinde çalışan,
              okunabilir ve modern <strong className="text-foreground">C++17</strong> ile yazılmış yüksek
              performanslı bir matris ve tensör işlem kütüphanesidir. Kütüphanenin temel felsefesi; karmaşık
              CUDA kernel başlatma, bellek tahsis ve senkronizasyon detaylarını kullanıcıdan soyutlayarak sade
              ve akıcı bir C++ arayüzü sunmaktır.
            </>
          ) : (
            <>
              <strong className="text-foreground">MatrixFlash-Pro</strong> is a high-performance matrix and tensor
              computation library designed for NVIDIA GPUs and written in modern <strong className="text-foreground">C++17</strong>.
              Its primary philosophy is to eliminate low-level CUDA boilerplate, memory management traps, and kernel
              launch overhead behind a fluent, developer-friendly C++ interface.
            </>
          )}
        </p>

        <div className="grid grid-cols-1 gap-4 sm:grid-cols-3 my-6">
          <div className="rounded-xl border border-border/80 bg-card/60 p-4">
            <Cpu className="size-5 text-primary mb-2" />
            <h4 className="font-mono text-sm font-bold text-foreground">
              {t('cuBLAS & TF32 GEMM', 'cuBLAS & TF32 GEMM')}
            </h4>
            <p className="mt-1 text-xs text-muted-foreground">
              {t(
                'Matris çarpımı NVIDIA cuBLAS ile donanım sınırlarında TF32 Tensor Core desteğiyle çalışır.',
                'Matrix multiplication utilizes NVIDIA cuBLAS with TF32 Tensor Core hardware acceleration.',
              )}
            </p>
          </div>

          <div className="rounded-xl border border-border/80 bg-card/60 p-4">
            <Zap className="size-5 text-accent mb-2" />
            <h4 className="font-mono text-sm font-bold text-foreground">
              {t('Autograd Ters-Mod Tape', 'Autograd Reverse-Mode Tape')}
            </h4>
            <p className="mt-1 text-xs text-muted-foreground">
              {t(
                'PyTorch benzeri hesaplama bandı ile otomatik türev ve GPU tabanlı geri yayılım (backprop).',
                'Automatic differentiation tape enabling reverse-mode backpropagation natively in VRAM.',
              )}
            </p>
          </div>

          <div className="rounded-xl border border-border/80 bg-card/60 p-4">
            <Layers className="size-5 text-emerald-400 mb-2" />
            <h4 className="font-mono text-sm font-bold text-foreground">
              {t('cuSOLVER & Sparse CSR', 'cuSOLVER & Sparse CSR')}
            </h4>
            <p className="mt-1 text-xs text-muted-foreground">
              {t(
                'SVD, QR, Cholesky, özdeğer analizleri ve seyrek (sparse) matris işlemleri tek çatı altında.',
                'SVD, QR, Cholesky, eigenvalue solvers, and sparse CSR matrices under one unified umbrella.',
              )}
            </p>
          </div>
        </div>

        <Callout type="tip" title={t('Hızlı Çıkarım: Tek Şemsiye Başlık', 'Pro Tip: Umbrella Header')}>
          {t(
            'Tüm modülleri tek seferde kullanmak için projenize #include "matrix_pro/matrix_pro.hpp" eklemeniz yeterlidir. Büyük projelerde derleme sürelerini düşürmek için alt modül başlıkları (örneğin matrix_pro/ops/product.hpp) ayrı ayrı da dahil edilebilir.',
            'Include #include "matrix_pro/matrix_pro.hpp" to access every module at once. In large codebases, include granular headers (e.g. matrix_pro/ops/product.hpp) to minimize compilation overhead.',
          )}
        </Callout>
      </DocSection>

      {/* Bölüm 2: Gereksinimler & Kurulum */}
      <DocSection id="kurulum" title={t('Gereksinimler & Kurulum', 'Prerequisites & Build Guide')}>
        <p>
          {t(
            'Kütüphaneyi kaynak koddan derlemek ve GPU üzerinde çalıştırmak için aşağıdaki bileşenler gereklidir:',
            'The following prerequisites are required to build and run MatrixFlash-Pro on your system:',
          )}
        </p>

        <ul className="ml-5 list-disc space-y-2 text-sm text-muted-foreground">
          <li>
            <strong className="text-foreground">NVIDIA GPU</strong>: Compute Capability 7.0+ (Volta, Turing, Ampere, Ada Lovelace, Hopper).
          </li>
          <li>
            <strong className="text-foreground">NVIDIA CUDA Toolkit</strong>: Sürüm 12.0 veya üzeri (12.3+ önerilir). <InlineCode>nvcc</InlineCode>, cuBLAS ve cuSOLVER kütüphanelerini içerir.
          </li>
          <li>
            <strong className="text-foreground">CMake</strong>: Sürüm 3.18 veya üzeri.
          </li>
          <li>
            <strong className="text-foreground">C++17 Derleyicisi</strong>: MSVC (Visual Studio 2019+), GCC 9+, ya da Clang 11+.
          </li>
        </ul>

        <SubHeading>{t('1. Depoyu Klonlayın', '1. Clone Repository')}</SubHeading>
        <CodeBlock code={cloneCode} filename="clone.sh" />

        <SubHeading>{t('2. CMake ile Yapılandırma ve Derleme', '2. Configure & Build with CMake')}</SubHeading>
        <CodeBlock code={cmakeBuildCode} filename="build.sh" />

        <SubHeading>{t('3. Testler ve Doğrulama', '3. Run Tests and Benchmarks')}</SubHeading>
        <CodeBlock code={runTestsCode} filename="run.ps1" />
      </DocSection>

      {/* Bölüm 3: Hızlı Başlangıç */}
      <DocSection id="hizli-baslangic" title={t('Hızlı Başlangıç (Quickstart)', 'Quickstart Guide')}>
        <p>
          {t(
            'Aşağıdaki minimal C++ programı; iki matrisi tanımlar, GPU üzerinde cuBLAS hızlandırmasıyla çarpar, ReLU aktivasyonunu zincirler ve güvenli şekilde host tarafına indirir:',
            'The following minimal C++ application allocates two matrices, performs cuBLAS-accelerated GEMM, chains ReLU non-linearity, and safely downloads the result:',
          )}
        </p>
        <CodeBlock
          code={`#include "matrix_pro/matrix_pro.hpp"
#include <iostream>

int main() {
    using matrix_pro::Matrix;

    // 1. Matrisleri tanımlayın (otomatik GPU'ya taşınır)
    const Matrix a{{1.0f, -2.0f}, {3.0f, 4.0f}};
    const Matrix b = Matrix::identity(2);

    // 2. GPU üzerinde zincirleme işlem: (A * B).relu()
    Matrix c = (a * b).relu();

    // 3. Sonucu host tarafına indirin
    c.download();

    std::cout << "Sonuc c(0,0): " << c.at(0, 0) << "\\n";
    std::cout << "Sonuc c(0,1): " << c.at(0, 1) << " (ReLU ile sifirlandi)\\n";
    return 0;
}`}
          filename="quickstart_minimal.cpp"
        />
      </DocSection>

      {/* Bölüm 4: Fonksiyon Çağrı Sırası & Yaşam Döngüsü */}
      <DocSection id="yasam-dongusu" title={t('Fonksiyon Çağrı Sırası & Yaşam Döngüsü', 'Calling Sequence & Lifecycle Guide')}>
        <p>
          {lang === 'tr' ? (
            <>
              MatrixFlash-Pro ile yüksek performanslı kod yazarken fonksiyonların hangi mantıksal sırada
              çağrılacağını bilmek hem hız hem de bellek güvenliği için kritiktir. Kütüphanedeki standart
              çalışma akışı şu 5 ana adımdan oluşur:
            </>
          ) : (
            <>
              Understanding the exact calling sequence in MatrixFlash-Pro is essential for peak throughput
              and zero-overhead memory safety. The standard execution lifecycle follows these 5 primary steps:
            </>
          )}
        </p>

        <div className="my-6 space-y-4">
          <div className="rounded-xl border border-border/80 bg-card/40 p-5">
            <h4 className="font-mono text-sm font-bold text-primary flex items-center gap-2">
              <span className="flex size-6 items-center justify-center rounded-full bg-primary/20 text-xs">1</span>
              {t('Tahsis (Allocation) & Bellek Modu', 'Allocation & Memory Mode')}
            </h4>
            <p className="mt-2 text-xs text-muted-foreground leading-relaxed">
              {t(
                'Büyük veri yüklerinde MemoryMode::device_only kullanarak gereksiz host kopyalamalarını engelleyin. Standart mod host_and_device ise testlerde pratik veri yükleme için uygundur.',
                'For compute-intensive workflows, specify MemoryMode::device_only to skip redundant host RAM mirrors. The default host_and_device mode is ideal for quick prototyping.',
              )}
            </p>
          </div>

          <div className="rounded-xl border border-border/80 bg-card/40 p-5">
            <h4 className="font-mono text-sm font-bold text-primary flex items-center gap-2">
              <span className="flex size-6 items-center justify-center rounded-full bg-primary/20 text-xs">2</span>
              {t('Veri Yükleme veya Cihaz İçi Üretim (Population)', 'Population & On-Device Generation')}
            </h4>
            <p className="mt-2 text-xs text-muted-foreground leading-relaxed">
              {t(
                'Host verisi varsa upload() çağrılır. Ancak mümkün olduğunca Matrix::randn_gpu, Matrix::uniform_gpu veya Matrix::zeros gibi doğrudan GPU üzerinde üretim yapan fabrikaları tercih edin.',
                'If importing host vectors, invoke upload(). However, prefer on-device factories such as Matrix::randn_gpu, Matrix::uniform_gpu, or Matrix::zeros to avoid PCIe transfers altogether.',
              )}
            </p>
          </div>

          <div className="rounded-xl border border-border/80 bg-card/40 p-5">
            <h4 className="font-mono text-sm font-bold text-primary flex items-center gap-2">
              <span className="flex size-6 items-center justify-center rounded-full bg-primary/20 text-xs">3</span>
              {t('Hesaplama & Zincirleme (Compute & Fused/In-Place)', 'Compute Pipeline & Fused Chains')}
            </h4>
            <p className="mt-2 text-xs text-muted-foreground leading-relaxed">
              {t(
                'İşlemleri zincirleyin ((a * b).relu()). Bellek tasarrufu gereken döngülerde in-place metodları (relu_(), add_()) veya birleşik çekirdekleri (fused_bias_gelu()) kullanarak global bellek turlarını azaltın.',
                'Chain operations ((a * b).relu()). In tight training loops, deploy in-place operators (relu_(), add_()) or fused kernels (fused_bias_gelu()) to cut global memory passes in half.',
              )}
            </p>
          </div>

          <div className="rounded-xl border border-border/80 bg-card/40 p-5">
            <h4 className="font-mono text-sm font-bold text-primary flex items-center gap-2">
              <span className="flex size-6 items-center justify-center rounded-full bg-primary/20 text-xs">4</span>
              {t('Asenkron Stream İşlemleri (Async Stream Reductions)', 'Async Stream Reductions')}
            </h4>
            <p className="mt-2 text-xs text-muted-foreground leading-relaxed">
              {t(
                'Eş zamanlı hesaplamalar için pool_stream(slot) kullanın. argmax_async ile GPU indirgemesini pinned host bellek yuvasına yazdırarak CPU döngüsünün bloklanmasını önleyin.',
                'Use pool_stream(slot) for concurrent GPU tasks. Post reduction results via argmax_async to pinned host memory slots without stalling the CPU thread.',
              )}
            </p>
          </div>

          <div className="rounded-xl border border-border/80 bg-card/40 p-5">
            <h4 className="font-mono text-sm font-bold text-primary flex items-center gap-2">
              <span className="flex size-6 items-center justify-center rounded-full bg-primary/20 text-xs">5</span>
              {t('Güvenli İndirme (Safe Download & Fail-Fast)', 'Safe Download & Fail-Fast Synchronization')}
            </h4>
            <p className="mt-2 text-xs text-muted-foreground leading-relaxed">
              {t(
                'GPU üzerinde yazılan bir matrisin elemanlarına at() veya data() ile erişmeden önce download() çağrılması ZORUNLUDUR. Kütüphane bayat veri okunmasına izin vermez ve istisna fırlatır.',
                'Invoking download() is MANDATORY before calling at() or data() on modified matrices. The library prevents silent stale reads by throwing a fail-fast runtime exception.',
              )}
            </p>
          </div>
        </div>

        <SubHeading>{t('Uçtan Uca Yaşam Döngüsü C++ Kodu', 'End-to-End Lifecycle C++ Example')}</SubHeading>
        <CodeBlock code={lifecycleCode} filename="lifecycle_demo.cpp" />
      </DocSection>

      {/* Bölüm 5: Autograd Yaşam Döngüsü */}
      <DocSection id="autograd-dongusu" title={t('Autograd Yaşam Döngüsü (Ters-Mod)', 'Autograd Tape Lifecycle')}>
        <p>
          {lang === 'tr' ? (
            <>
              Yapay sinir ağı katmanları eğitilirken, değişkenler <InlineCode>Variable</InlineCode> ile sarılır.
              İleri yayılım esnasında hesaplama grafiği (tape) dinamik olarak inşa edilir ve <InlineCode>backward()</InlineCode>{' '}
              çağrısıyla zincir kuralı geriye doğru işletilir.
            </>
          ) : (
            <>
              When training neural network layers, tensors are wrapped in <InlineCode>Variable</InlineCode> nodes.
              Forward operations automatically record dependencies on the tape, and <InlineCode>backward()</InlineCode>{' '}
              propagates gradients backwards according to the chain rule.
            </>
          )}
        </p>

        <CodeBlock code={autogradLifecycleCode} filename="autograd_training.cpp" />
      </DocSection>

      {/* Bölüm 6: İstisna & Hata Yönetimi */}
      <DocSection id="hata-yonetimi" title={t('İstisna & Hata Yönetimi Hiyerarşisi', 'Exception Hierarchy')}>
        <p>
          {lang === 'tr' ? (
            <>
              MatrixFlash-Pro’da fırlatılan tüm istisnalar <InlineCode>matrix_pro::MatrixProError</InlineCode>{' '}
              taban sınıfından türer (<InlineCode>std::runtime_error</InlineCode> mirasçısıdır). Bu sayede tek bir{' '}
              <InlineCode>catch (const MatrixProError&amp;)</InlineCode> bloğu ile tüm hataları yakalayabilir veya
              alt sınıflarla nedene özel hata yönetimi yapabilirsiniz:
            </>
          ) : (
            <>
              Every exception thrown across MatrixFlash-Pro derives from <InlineCode>matrix_pro::MatrixProError</InlineCode>{' '}
              (which inherits from <InlineCode>std::runtime_error</InlineCode>). A single catch block handles all library errors,
              or you can catch granular subclasses for targeted handling:
            </>
          )}
        </p>

        <div className="divide-y divide-border/60 rounded-xl border border-border/80 overflow-hidden text-xs font-mono my-5">
          <div className="grid grid-cols-1 gap-2 bg-card/60 p-3 sm:grid-cols-3">
            <span className="text-primary font-bold">ShapeMismatchError</span>
            <span className="sm:col-span-2 text-muted-foreground font-sans">
              {t('Matris veya tensör boyutları çarpım veya toplama için uyuşmuyor.', 'Matrix or tensor dimensions are incompatible for operation.')}
            </span>
          </div>
          <div className="grid grid-cols-1 gap-2 bg-card/40 p-3 sm:grid-cols-3">
            <span className="text-primary font-bold">InvalidArgumentError</span>
            <span className="sm:col-span-2 text-muted-foreground font-sans">
              {t('Geçersiz argüman (negatif boyut, geçersiz dropout olasılığı vb.).', 'Invalid argument passed (negative shape, bad probability, etc.).')}
            </span>
          </div>
          <div className="grid grid-cols-1 gap-2 bg-card/60 p-3 sm:grid-cols-3">
            <span className="text-primary font-bold">OutOfRangeError</span>
            <span className="sm:col-span-2 text-muted-foreground font-sans">
              {t('Dilimleme veya indeks sınır dışına taştı (at, slice, gather).', 'Slice or index escaped container bounds (at, slice, gather).')}
            </span>
          </div>
          <div className="grid grid-cols-1 gap-2 bg-card/40 p-3 sm:grid-cols-3">
            <span className="text-primary font-bold">CudaError</span>
            <span className="sm:col-span-2 text-muted-foreground font-sans">
              {t('CUDA Runtime, cuBLAS veya cuDNN çağrısı başarısız oldu.', 'A CUDA runtime, cuBLAS, or cuDNN driver call failed.')}
            </span>
          </div>
          <div className="grid grid-cols-1 gap-2 bg-card/60 p-3 sm:grid-cols-3">
            <span className="text-primary font-bold">SolverError</span>
            <span className="sm:col-span-2 text-muted-foreground font-sans">
              {t('cuSOLVER ayrışımı (SVD, Cholesky, Eigen) yakınsamadı veya tekil.', 'cuSOLVER factorization (SVD, Cholesky, Eigen) failed or matrix is singular.')}
            </span>
          </div>
          <div className="grid grid-cols-1 gap-2 bg-card/40 p-3 sm:grid-cols-3">
            <span className="text-primary font-bold">IoError</span>
            <span className="sm:col-span-2 text-muted-foreground font-sans">
              {t('Dosya kaydetme (save) veya yükleme (load) başarısız oldu.', 'Disk serialization or deserialization (save/load) failed.')}
            </span>
          </div>
        </div>

        <div className="mt-8 flex items-center justify-between border-t border-border/80 pt-6">
          <Link
            href="/docs/matris-nedir"
            className="inline-flex items-center gap-2 text-sm font-semibold text-primary hover:underline"
          >
            {t('Sonraki: Matris Dersleri', 'Next: Matrix Mathematics Course')}
            <ArrowRight className="size-4" />
          </Link>
        </div>
      </DocSection>
    </article>
  )
}
