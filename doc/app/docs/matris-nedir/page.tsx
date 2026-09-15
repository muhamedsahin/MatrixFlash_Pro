import { Cpu, Layers, Zap } from 'lucide-react'
import { CodeBlock } from '../../../components/code-block'
import { Math } from '../../../components/math'
import { MatrixFigure } from '../../../components/docs/matrix-figure'
import {
  Callout,
  DocSection,
  InlineCode,
  PageHeader,
  SubHeading,
} from '../../../components/docs/doc-ui'

const matmulCode = `Matrix a{{1.0f, 2.0f, 3.0f},
         {4.0f, 5.0f, 6.0f}};        // 2x3

Matrix b{{7.0f,  8.0f},
         {9.0f,  10.0f},
         {11.0f, 12.0f}};            // 3x2

Matrix c = a * b;                    // 2x2 sonuç
c.download();
c.print();`

const chainCode = `// Her operasyon yeni bir Matrix döndürür => zincirlenebilir
Matrix result = input
    .transpose()      // devrik
    .scale(0.5f)      // skaler çarpım
    .relu()           // aktivasyon
    .softmax();       // olasılık dağılımı

result.download();    // yalnızca burada host'a iner`

export default function MatrisNedirPage() {
  return (
    <article>
      <PageHeader
        eyebrow="Temeller"
        title="Matris Nedir?"
        description="Matrislerin matematiksel tanımı, temel işlemleri ve bu işlemlerin neden GPU üzerinde bu kadar hızlı çalıştığını adım adım inceleyelim."
      />

      <DocSection title="Tanım">
        <p>
          Bir <strong className="text-foreground">matris</strong>, sayıların
          satır ve sütunlar halinde dikdörtgen bir tabloda düzenlenmiş halidir.{' '}
          <Math>{'m \\times n'}</Math> boyutundaki bir matris,{' '}
          <Math>{'m'}</Math> satır ve <Math>{'n'}</Math> sütundan oluşur.
        </p>
        <Math display>
          {'A = \\begin{bmatrix} a_{11} & a_{12} & a_{13} \\\\ a_{21} & a_{22} & a_{23} \\end{bmatrix}'}
        </Math>
        <p>
          Yukarıdaki <Math>{'A'}</Math> matrisi <Math>{'2 \\times 3'}</Math>{' '}
          boyutundadır. Her bir <Math>{'a_{ij}'}</Math> elemanı,{' '}
          <Math>{'i'}</Math>. satır ve <Math>{'j'}</Math>. sütundaki değeri
          temsil eder. MatrixFlash-Pro içinde bu değerler{' '}
          <InlineCode>float</InlineCode> tipinde saklanır.
        </p>
        <MatrixFigure
          data={[
            ['a₁₁', 'a₁₂', 'a₁₃'],
            ['a₂₁', 'a₂₂', 'a₂₃'],
          ]}
          highlight={[[1, 1]]}
          caption="a₂₂ elemanı → 2. satır, 2. sütun"
        />
      </DocSection>

      <DocSection title="Temel İşlemler">
        <SubHeading>Toplama</SubHeading>
        <p>
          Aynı boyuttaki iki matris, karşılıklı elemanları toplanarak
          (elementwise) toplanır:
        </p>
        <Math display>
          {'(A + B)_{ij} = A_{ij} + B_{ij}'}
        </Math>
        <div className="flex flex-wrap items-center justify-center gap-4">
          <MatrixFigure data={[['1', '2'], ['3', '4']]} />
          <span className="font-mono text-2xl text-primary">+</span>
          <MatrixFigure data={[['5', '6'], ['7', '8']]} accent="accent" />
          <span className="font-mono text-2xl text-primary">=</span>
          <MatrixFigure
            data={[['6', '8'], ['10', '12']]}
            accent="chart-3"
          />
        </div>

        <SubHeading>Skaler Çarpım</SubHeading>
        <p>
          Bir matrisin her elemanı bir <Math>{'c'}</Math> skaleri ile çarpılır.
          Kütüphanede bu <InlineCode>scale(c)</InlineCode> ile yapılır:
        </p>
        <Math display>{'(cA)_{ij} = c \\cdot A_{ij}'}</Math>

        <SubHeading>Devrik (Transpose)</SubHeading>
        <p>
          Satırlar ile sütunların yer değiştirmesidir.{' '}
          <Math>{'m \\times n'}</Math> matris, <Math>{'n \\times m'}</Math>{' '}
          matrise dönüşür:
        </p>
        <Math display>{'(A^{T})_{ij} = A_{ji}'}</Math>
        <div className="flex flex-wrap items-center justify-center gap-4">
          <MatrixFigure
            data={[['1', '2', '3'], ['4', '5', '6']]}
            caption="A (2×3)"
          />
          <span className="font-mono text-2xl text-primary">→</span>
          <MatrixFigure
            data={[['1', '4'], ['2', '5'], ['3', '6']]}
            accent="chart-3"
            caption="Aᵀ (3×2)"
          />
        </div>
      </DocSection>

      <DocSection title="Matris Çarpımı">
        <p>
          Matris çarpımı, matrislerin en güçlü ve en maliyetli işlemidir.{' '}
          <Math>{'A'}</Math> matrisinin <Math>{'m \\times k'}</Math>,{' '}
          <Math>{'B'}</Math> matrisinin ise <Math>{'k \\times n'}</Math>{' '}
          boyutunda olması gerekir. Sonuç <Math>{'m \\times n'}</Math>{' '}
          boyutundadır:
        </p>
        <Math display>
          {'C_{ij} = \\sum_{p=1}^{k} A_{ip} \\cdot B_{pj}'}
        </Math>
        <p>
          Yani sonuç matrisinin her bir elemanı, <Math>{'A'}</Math>&apos;nın bir
          satırı ile <Math>{'B'}</Math>&apos;nin bir sütununun{' '}
          <strong className="text-foreground">nokta çarpımıdır</strong>. Bu
          işlem, MatrixFlash-Pro içinde <InlineCode>*</InlineCode> operatörü ile
          çağrılır ve arka planda shared-memory tiled CUDA kernel tarafından
          yürütülür.
        </p>
        <CodeBlock code={matmulCode} filename="matmul.cpp" />
        <Callout type="tip" title="Neden tiled CUDA kernel?">
          Bir <Math>{'n \\times n'}</Math> matris çarpımı yaklaşık{' '}
          <Math>{'n^{3}'}</Math> işlem gerektirir. Tiled yaklaşım, veriyi shared
          memory bloklarıyla yeniden kullanarak global bellek erişimini azaltır.
        </Callout>
      </DocSection>

      <DocSection id="gpu" title="Neden GPU? — Paralellik Mantığı">
        <p>
          Matris işlemlerinin çoğu{' '}
          <strong className="text-foreground">
            birbirinden bağımsız
          </strong>{' '}
          hesaplamalardır. Örneğin bir toplama işleminde her elemanın toplamı,
          diğer elemanlardan bağımsız olarak hesaplanabilir. İşte GPU tam olarak
          bunda üstündür.
        </p>
        <div className="my-6 grid gap-4 sm:grid-cols-3">
          {[
            {
              icon: Cpu,
              title: 'Binlerce çekirdek',
              body: 'GPU, aynı anda binlerce eleman üzerinde işlem yapabilir.',
            },
            {
              icon: Zap,
              title: 'SIMT modeli',
              body: 'Tek komut, çok sayıda veri üzerinde eş zamanlı çalışır.',
            },
            {
              icon: Layers,
              title: 'Yüksek bant genişliği',
              body: 'Device belleği, büyük matrislere hızlı erişim sağlar.',
            },
          ].map((c) => (
            <div
              key={c.title}
              className="rounded-xl border border-border bg-card p-5"
            >
              <c.icon className="size-5 text-primary" />
              <p className="mt-3 font-semibold text-foreground">{c.title}</p>
              <p className="mt-1 text-sm text-muted-foreground">{c.body}</p>
            </div>
          ))}
        </div>
        <p>
          MatrixFlash-Pro, verinizi bir kez GPU belleğine (device memory) yükler
          ve tüm ara işlemleri orada tutar. Yavaş olan host↔device veri transferi
          yalnızca gerektiğinde yapılır.
        </p>
      </DocSection>

      <DocSection id="zincirleme" title="Zincirleme (Chaining) API">
        <p>
          Kütüphanedeki hemen her operasyon yeni bir{' '}
          <InlineCode>Matrix</InlineCode> döndürür. Bu tasarım, işlemleri akıcı
          bir şekilde zincirlemenizi sağlar. Ara sonuçlar device belleğinde kalır
          ve yalnızca <InlineCode>download()</InlineCode> ile host&apos;a iner:
        </p>
        <CodeBlock code={chainCode} filename="chaining.cpp" />
        <Callout type="info" title="Özet">
          Matris = sayıların ızgarası. GPU = bu ızgaranın her hücresini paralel
          işleyen motor. MatrixFlash-Pro = bu ikisini sade bir C++ arayüzünde
          birleştiren köprü.
        </Callout>
      </DocSection>
    </article>
  )
}
