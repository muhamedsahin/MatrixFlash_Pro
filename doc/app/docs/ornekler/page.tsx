'use client'

import React from 'react'
import Link from 'next/link'
import { CodeBlock } from '@/components/code-block'
import {
  Callout,
  DocSection,
  InlineCode,
  PageHeader,
} from '@/components/docs/doc-ui'
import { useLanguage } from '@/lib/language-context'

const exBasic = `#include "matrix_pro/matrix_pro.hpp"
#include <iostream>

int main() {
    using matrix_pro::Matrix;

    // 1. Host verisinden 2x2 matris oluştur (otomatik GPU'ya taşınır)
    const Matrix a{{1.0f, 2.0f},
                   {3.0f, 4.0f}};
    const Matrix b = Matrix::identity(2);

    // 2. cuBLAS GEMM ile çarpım ve ReLU aktivasyonu zincirleme
    Matrix c = (a * b).relu();

    // 3. Fail-fast kuralı: at() öncesinde download() zorunludur
    c.download();
    std::cout << "Sonuc c(0,0): " << c.at(0, 0) << "\\n";
    std::cout << "Sonuc c(1,1): " << c.at(1, 1) << "\\n";

    return 0;
}`

const exHighPerf = `#include "matrix_pro/matrix_pro.hpp"
#include <iostream>

int main() {
    using namespace matrix_pro;

    const std::size_t size = 1024;

    // 1. Host mirror olmadan device-only bellek tahsisi (0 PCIe gecikmesi)
    Matrix x(size, size, MemoryMode::device_only);
    Matrix w(size, size, MemoryMode::device_only);
    x.fill(1.0f);
    w.fill(0.5f);

    // 2. cuBLAS SGEMM + TF32 Tensor Core ile matris çarpımı
    Matrix prod = x * w;

    // 3. Fused kernel zinciri: bias ekleme ve GeLU tek kernel'da birleşir
    Matrix bias = Matrix::zeros(size, size);
    Matrix activated = fused_bias_gelu(prod, bias);

    // 4. In-place (yerinde) operasyon ile 0 ek bellek tahsisi
    relu_(activated);

    // 5. Host tarafına indir ve doğrula
    activated.download();
    std::cout << "Ortalama deger: " << activated.mean() << "\\n";

    return 0;
}`

const exAutograd = `#include "matrix_pro/matrix_pro.hpp"
#include <iostream>

int main() {
    using namespace matrix_pro;

    // 1. Öğrenilebilir parametreler (requires_grad = true)
    Variable x(Matrix::randn(64, 128), /*requires_grad=*/false);
    Variable w1(Matrix::glorot(128, 64), /*requires_grad=*/true);
    Variable b1(Matrix::zeros(1, 64),   /*requires_grad=*/true);
    Variable w2(Matrix::glorot(64, 10),  /*requires_grad=*/true);
    Variable b2(Matrix::zeros(1, 10),   /*requires_grad=*/true);

    Variable target(Matrix::ones(64, 10), /*requires_grad=*/false);

    // Basit bir eğitim adımı (Forward + Backward + SGD)
    for (int epoch = 0; epoch < 5; ++epoch) {
        // İleri yayılım (Forward pass)
        Variable h1 = x.matmul(w1).broadcast_add(b1).relu();
        Variable logits = h1.matmul(w2).broadcast_add(b2);
        Variable loss = mse_loss(logits, target);

        std::cout << "Epoch " << epoch << " | Loss: " << loss.value().mean() << "\\n";

        // Geri yayılım (Backward pass - gradyanlar GPU üzerinde hesaplanır)
        loss.backward();

        // Parametreleri güncelle (SGD: w = w - lr * grad)
        const float lr = 0.01f;
        // Gradyanları sıfırla
        w1.zero_grad();
        b1.zero_grad();
        w2.zero_grad();
        b2.zero_grad();
    }

    return 0;
}`

const exCNN = `#include "matrix_pro/matrix_pro.hpp"
#include <iostream>

int main() {
    using namespace matrix_pro;

    // Girdi Tensörü: [Batch=8, Kanal=3, Yükseklik=32, Genişlik=32]
    Tensor input({8, 3, 32, 32}, MemoryMode::device_only);
    input.fill(1.0f);

    // 16 adet 3x3x3 konvolüsyon filtresi
    Tensor weights({16, 3, 3, 3}, MemoryMode::device_only);
    weights.fill(0.1f);
    Tensor bias({16}, MemoryMode::device_only);
    bias.fill(0.01f);

    // 2B Konvolüsyon (stride=1, padding=1)
    Tensor conv_out = conv2d(input, weights, bias, /*stride=*/1, /*padding=*/1);

    // 2x2 Max Pooling (stride=2)
    Tensor pooled = max_pool2d(conv_out, /*kernel_size=*/2, /*stride=*/2);

    std::cout << "Girdi Rank: " << input.rank() << " | Havuzlama cikis boyutu: "
              << pooled.shape()[0] << "x" << pooled.shape()[1] << "x"
              << pooled.shape()[2] << "x" << pooled.shape()[3] << "\\n";

    return 0;
}`

const exSparse = `#include "matrix_pro/matrix_pro.hpp"
#include <iostream>

int main() {
    using namespace matrix_pro;

    // Seyrek büyük bir matris oluştur
    Matrix dense = Matrix::zeros(5000, 5000);
    dense.at(10, 20) = 4.5f;
    dense.at(100, 200) = 8.2f;
    dense.upload();

    // CSR (Compressed Sparse Row) formatına dönüştür
    SparseCSR sparse = SparseCSR::from_dense(dense, /*threshold=*/1e-5f);
    std::cout << "Sifir olmayan eleman sayisi (nnz): " << sparse.nnz() << "\\n";
    std::cout << "Seyreklik orani: %" << sparse.sparsity() * 100.0f << "\\n";

    // Seyrek matris-vektör çarpımı (SpMV)
    Matrix x = Matrix::ones(5000, 1);
    Matrix y = spmv(sparse, x);

    y.download();
    std::cout << "SpMV Sonuc y(10, 0): " << y.at(10, 0) << "\\n";
    return 0;
}`

const exView = `#include "matrix_pro/matrix_pro.hpp"
#include <iostream>

int main() {
    using namespace matrix_pro;

    Matrix m = Matrix::randn(1024, 2048);

    // 1. Sıfır kopyalama transpoz (strides takas edilir)
    MatrixView transposed = transpose_view(m); // 2048 x 1024
    std::cout << "Transpoz boyutu: " << transposed.rows() << "x" << transposed.cols() << "\\n";

    // 2. Sıfır kopyalama alt matris dilimi (slice)
    MatrixView patch = slice_view(m, 0, 64, 0, 64);
    std::cout << "Dilim boyutu: " << patch.rows() << "x" << patch.cols() << "\\n";

    // 3. İhtiyaç duyulduğunda yoğun Matrix nesnesine dönüştürün
    Matrix dense_patch = materialize(patch);
    dense_patch.download();
    std::cout << "Dense patch ortalamasi: " << dense_patch.mean() << "\\n";

    return 0;
}`

const exStreams = `#include "matrix_pro/matrix_pro.hpp"
#include <iostream>

int main() {
    using namespace matrix_pro;

    Matrix m = Matrix::randn(2048, 2048);

    // Pinned host bellek yuvası tahsis et
    std::size_t* host_slot = static_cast<std::size_t*>(pin_host(sizeof(std::size_t)));

    // CPU'yu bloklamayan asenkron argmax indirgemesi
    argmax_async(m, host_slot, [](std::size_t best_idx) {
        std::cout << "Asenkron Callback: En buyuk indeks -> " << best_idx << "\\n";
    });

    std::cout << "Ana CPU thread kesintisiz calismaya devam ediyor...\\n";

    // Tüm stream havuzunu senkronize et
    synchronize_pool();
    unpin_host(host_slot);

    return 0;
}`

export default function OrneklerPage() {
  const { lang, t } = useLanguage()

  const examples = [
    {
      id: 'basic',
      title: { tr: '1. Temel GPU Kullanımı & Zincirleme', en: '1. Basic GPU Usage & Chaining' },
      desc: {
        tr: 'cuBLAS matris çarpımı, ReLU aktivasyonu ve güvenli host indirme (download) mantığını gösteren giriş örneği.',
        en: 'Introductory example showcasing cuBLAS GEMM, ReLU activation chaining, and safe host download.',
      },
      code: exBasic,
      tags: ['cuBLAS', 'GEMM', 'relu', 'download'],
    },
    {
      id: 'high-perf',
      title: { tr: '2. Yüksek Performans & Fused / In-Place Kernel’lar', en: '2. High Performance Fused & In-Place Pipelines' },
      desc: {
        tr: 'MemoryMode::device_only ile host mirror gecikmesini sıfırlayan, fused_bias_gelu ve relu_ ile bellek bant genişliğini doyuran kod parçası.',
        en: 'Device-only allocation avoiding host mirror overhead, coupled with fused_bias_gelu and relu_ for peak memory bandwidth.',
      },
      code: exHighPerf,
      tags: ['device_only', 'fused_bias_gelu', 'relu_', 'TF32'],
    },
    {
      id: 'autograd-mlp',
      title: { tr: '3. Autograd ile Çok Katmanlı Algılayıcı (MLP) Eğitimi', en: '3. MLP Training with Autograd Reverse-Mode Tape' },
      desc: {
        tr: 'Variable hesaplama bandı, ileri yayılım, geriye yayılım (loss.backward()) ve parametre güncellemelerini içeren sinir ağı eğitim döngüsü.',
        en: 'Neural network training loop featuring Variable tape, forward pass, loss.backward(), and gradient inspection.',
      },
      code: exAutograd,
      tags: ['Variable', 'backward', 'mse_loss', 'zero_grad'],
    },
    {
      id: 'cnn',
      title: { tr: '4. 2D Konvolüsyon ve Havuzlama (CNN)', en: '4. 2D Convolution & Pooling (CNN)' },
      desc: {
        tr: 'Rank-4 Tensor kabı üzerinde NCHW konvolüsyon ve 2x2 MaxPool katmanlarının uçtan uca uygulanışı.',
        en: 'End-to-end forward application of NCHW 2D convolution and 2x2 MaxPool layers over rank-4 Tensor objects.',
      },
      code: exCNN,
      tags: ['Tensor', 'conv2d', 'max_pool2d', 'NCHW'],
    },
    {
      id: 'sparse',
      title: { tr: '5. Seyrek Matrisler (Sparse CSR & SpMV)', en: '5. Sparse CSR Matrices & SpMV' },
      desc: {
        tr: 'Sıkıştırılmış satır (CSR) formatı oluşturma, seyreklik analizi ve seyrek matris-vektör çarpımı.',
        en: 'Constructing Compressed Sparse Row (CSR) buffers, analyzing sparsity, and executing fast SpMV.',
      },
      code: exSparse,
      tags: ['SparseCSR', 'spmv', 'sparsity'],
    },
    {
      id: 'view',
      title: { tr: '6. Sıfır Kopyalama Görünümleri (MatrixView)', en: '6. Zero-Copy Strided MatrixView' },
      desc: {
        tr: 'VRAM üzerinde tek bir bayt kopyalamadan adım (stride) manipülasyonu ile anında transpoz ve alt matris dilimleme.',
        en: 'Instant transpose and sub-matrix slicing via stride manipulation with zero VRAM copying overhead.',
      },
      code: exView,
      tags: ['MatrixView', 'transpose_view', 'slice_view', 'materialize'],
    },
    {
      id: 'streams',
      title: { tr: '7. Çoklu Stream Havuzu ve Asenkron İndirgemeler', en: '7. Multi-Stream Pool & Async Reductions' },
      desc: {
        tr: 'CPU ana iş parçacığını bloklamadan pinned belleğe sonuç yazan asenkron argmax ve stream havuzu senkronizasyonu.',
        en: 'Non-blocking asynchronous argmax reduction posting into pinned host memory via multi-stream pool.',
      },
      code: exStreams,
      tags: ['pool_stream', 'argmax_async', 'pin_host', 'non_blocking'],
    },
  ]

  return (
    <article className="space-y-6">
      <PageHeader
        eyebrow={t('KOD DEPOSU & ÖRNEKLER', 'CODE EXAMPLES')}
        title={t('Üretime Hazır C++17 Örnekleri', 'Production-Grade C++17 Examples')}
        description={t(
          'Gerçek dünya kullanım senaryolarında MatrixFlash-Pro kütüphanesini en yüksek verimle çalıştırmanızı sağlayacak, doğrudan kopyalayıp derleyebileceğiniz kod örnekleri.',
          'Ready-to-compile, production-grade C++17 examples demonstrating peak performance across diverse real-world workflows.',
        )}
      />

      <Callout type="tip" title={t('Derleme Talimatı', 'Build Instructions')}>
        {t(
          'Bu örneklerin her biri MatrixFlash-Pro umbrella başlığını kullanır. Kendi CMakeLists.txt dosyanızda target_link_libraries(hedef PRIVATE matrix_pro) ekleyerek doğrudan derleyebilirsiniz.',
          'Each example includes the umbrella header. In your own CMakeLists.txt, simply declare target_link_libraries(target PRIVATE matrix_pro) to build effortlessly.',
        )}
      </Callout>

      <div className="mt-8 space-y-12">
        {examples.map((ex) => (
          <DocSection key={ex.id} id={ex.id} title={ex.title[lang]}>
            <p className="text-sm text-muted-foreground">{ex.desc[lang]}</p>
            <div className="mb-4 flex flex-wrap gap-1.5">
              {ex.tags.map((tag) => (
                <span
                  key={tag}
                  className="rounded-md border border-border/80 bg-secondary/40 px-2 py-0.5 font-mono text-[11px] text-primary"
                >
                  #{tag}
                </span>
              ))}
            </div>
            <CodeBlock code={ex.code} filename={`${ex.id}.cpp`} />
          </DocSection>
        ))}
      </div>

      {/* GitHub CTA Box */}
      <div className="mt-12 rounded-2xl border border-primary/30 bg-gradient-to-b from-primary/[0.08] to-transparent p-8 text-center shadow-xl">
        <h3 className="font-mono text-lg font-bold text-foreground">
          {t('Daha Fazla Örnek & Test Kodu', 'Looking for More Examples & Benchmarks?')}
        </h3>
        <p className="mt-2 text-sm text-muted-foreground max-w-xl mx-auto">
          {t(
            'Deponun examples/ ve tests/ klasörlerinde kütüphanenin cuSOLVER, otograd ve CNN test senaryolarını detaylıca inceleyebilirsiniz.',
            'Explore the examples/ and tests/ directories in the GitHub repository for additional cuSOLVER and Autograd test suites.',
          )}
        </p>
        <a
          href="https://github.com/muhamedsahin/MatrixFlash_Pro"
          target="_blank"
          rel="noreferrer"
          className="mt-5 inline-flex items-center gap-2 rounded-xl bg-primary px-6 py-3 text-sm font-semibold text-primary-foreground transition-all hover:scale-105"
        >
          GitHub Repository
        </a>
      </div>
    </article>
  )
}
