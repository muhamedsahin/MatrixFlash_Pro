import Link from 'next/link'
import { ArrowRight } from 'lucide-react'
import { CodeBlock } from '../../components/code-block'
import {
  Callout,
  DocSection,
  InlineCode,
  PageHeader,
  SubHeading,
} from '../../components/docs/doc-ui'

const cloneCode = `# Depoyu klonla
git clone https://github.com/muhamedsahin/MatrixFlash_Pro.git
cd MatrixFlash_Pro`

const buildCode = `# 1. Build klasörü oluştur ve CMake ile yapılandır
cmake -B build -S .

# 2. Release modunda derle
cmake --build build --config Release`

const runCode = `# Testleri çalıştır
& .\\build\\Release\\matrix_pro_tests.exe

# Benchmark aracını 1024x1024 için çalıştır
& .\\build\\Release\\matrix_pro_benchmark.exe 1024

# Örnek programı çalıştır
& .\\build\\Release\\matrix_pro_example.exe`

const quickCode = `#include "matrix_pro/matrix.hpp"
#include <iostream>

using matrix_pro::Matrix;

int main() {
    // Host verisinden matris oluştur (otomatik olarak GPU'ya yüklenir)
    Matrix a{{1.0f, 2.0f},
             {3.0f, 4.0f}};

    Matrix b = Matrix::identity(2);

    // Zincirleme işlem: çarpım -> ReLU aktivasyonu
    Matrix c = (a * b).relu();

    // Sonucu host tarafına geri indir ve yazdır
    c.download();
    c.print();

    return 0;
}`

export default function DocsHomePage() {
  return (
    <article>
      <PageHeader
        eyebrow="Başlangıç"
        title="Genel Bakış"
        description="MatrixFlash-Pro, CUDA kernel karmaşıklığını soyutlayarak GPU üzerinde yüksek performanslı matris işlemleri sunan, modern C++17 ile yazılmış bir kütüphanedir."
      />

      <DocSection title="MatrixFlash-Pro nedir?">
        <p>
          Kütüphanenin temel amacı, GPU programlamanın düşük seviyeli
          detaylarıyla (bellek yönetimi, kernel başlatma, senkronizasyon)
          uğraşmadan, geliştiricilere <InlineCode>numpy</InlineCode> benzeri sade
          ve akıcı bir arayüz sunmaktır.
        </p>
        <p>
          Tüm hesaplama yükü NVIDIA GPU üzerinde yürütülür. Kritik matris
          çarpımı işlemleri shared-memory tiled CUDA kernel ile;
          elementwise operasyonlar, indirgemeler ve aktivasyon fonksiyonları özel
          olarak yazılmış CUDA kernelları ile gerçekleştirilir.
        </p>
        <Callout type="tip" title="Zincirleme (chaining) mantığı">
          İşlemler device belleğinde art arda zincirlenir. Veri, host tarafına
          yalnızca siz açıkça <InlineCode>download()</InlineCode> çağırdığınızda
          aktarılır. Bu sayede gereksiz host↔device transferi engellenir.
        </Callout>
      </DocSection>

      <DocSection id="gereksinimler" title="Gereksinimler">
        <p>Kütüphaneyi derlemek ve çalıştırmak için aşağıdakiler gereklidir:</p>
        <ul className="ml-5 list-disc space-y-2">
          <li>
            <strong className="text-foreground">CUDA destekli NVIDIA GPU</strong>{' '}
            — hesaplama bu donanım üzerinde yürütülür.
          </li>
          <li>
            <strong className="text-foreground">NVIDIA CUDA Toolkit</strong> —{' '}
            <InlineCode>nvcc</InlineCode> derleyicisini içerir.
          </li>
          <li>
            <strong className="text-foreground">CMake</strong> — sürüm 3.18 veya
            üzeri önerilir.
          </li>
          <li>
            <strong className="text-foreground">C++17</strong> uyumlu bir
            derleyici (MSVC, GCC veya Clang).
          </li>
        </ul>
        <Callout type="warn" title="GPU zorunludur">
          MatrixFlash-Pro tamamen GPU üzerinde çalışır. CUDA destekli bir NVIDIA
          ekran kartı olmadan kütüphane çalıştırılamaz.
        </Callout>
      </DocSection>

      <DocSection id="kurulum" title="Kurulum & Derleme">
        <p>Öncelikle depoyu klonlayın:</p>
        <CodeBlock code={cloneCode} lang="bash" />
        <p>
          Proje standart bir CMake yapısı kullanır. Aşağıdaki komutlarla
          yapılandırıp derleyebilirsiniz:
        </p>
        <CodeBlock code={buildCode} lang="bash" />
        <p>
          Derleme başarıyla tamamlandığında,{' '}
          <InlineCode>build/Release</InlineCode> klasöründe test, benchmark ve
          örnek çalıştırılabilir dosyaları oluşur:
        </p>
        <CodeBlock code={runCode} lang="bash" />
      </DocSection>

      <DocSection id="hizli-baslangic" title="Hızlı Başlangıç">
        <p>
          Aşağıdaki örnek, iki matris oluşturup çarpar, sonuca ReLU aktivasyonu
          uygular ve nihai sonucu ekrana yazdırır. Tüm hesaplama GPU üzerinde
          gerçekleşir:
        </p>
        <CodeBlock code={quickCode} filename="main.cpp" />
        <Callout type="info" title="Ne oldu?">
          <InlineCode>a</InlineCode> ve <InlineCode>b</InlineCode> oluşturulduğu
          anda GPU belleğine yüklendi. <InlineCode>{'(a * b).relu()'}</InlineCode>{' '}
          zinciri tamamen device üzerinde çalıştı. Veri host tarafına yalnızca{' '}
          <InlineCode>download()</InlineCode> ile geri getirildi.
        </Callout>
      </DocSection>

      <div className="mt-8 grid gap-4 sm:grid-cols-2">
        <Link
          href="/docs/matris-nedir"
          className="group rounded-xl border border-border bg-card p-5 transition-colors hover:border-primary/40"
        >
          <p className="font-mono text-xs uppercase tracking-widest text-muted-foreground">
            Sonraki
          </p>
          <p className="mt-2 flex items-center gap-2 font-semibold">
            Matris Nedir?
            <ArrowRight className="size-4 text-primary transition-transform group-hover:translate-x-1" />
          </p>
          <p className="mt-1 text-sm text-muted-foreground">
            Matrislerin matematiksel temelleri ve GPU üzerindeki mantığı.
          </p>
        </Link>
        <Link
          href="/docs/api"
          className="group rounded-xl border border-border bg-card p-5 transition-colors hover:border-primary/40"
        >
          <p className="font-mono text-xs uppercase tracking-widest text-muted-foreground">
            Referans
          </p>
          <p className="mt-2 flex items-center gap-2 font-semibold">
            API Referansı
            <ArrowRight className="size-4 text-primary transition-transform group-hover:translate-x-1" />
          </p>
          <p className="mt-1 text-sm text-muted-foreground">
            Tüm oluşturucular, operasyonlar ve fonksiyonlar tek tek.
          </p>
        </Link>
      </div>
    </article>
  )
}
