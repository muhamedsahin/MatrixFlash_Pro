import { CodeBlock } from '../../../components/code-block'
import {
  Callout,
  DocSection,
  InlineCode,
  PageHeader,
} from '../../../components/docs/doc-ui'

const linearLayer = `#include "matrix_pro/matrix.hpp"
using matrix_pro::Matrix;

// Basit bir tam bağlı (dense) katman: y = relu(x·W + b)
int main() {
    Matrix x = Matrix::randn(32, 128);   // 32 örnek, 128 özellik
    Matrix W = Matrix::randn(128, 64);   // ağırlıklar
    Matrix b{{/* 64 bias değeri */}};    // 1x64 bias

    Matrix y = (x * W).broadcast_add(b).relu();

    y.download();
    std::cout << "Cikti boyutu: 32x64, ortalama: " << y.mean() << "\\n";
    return 0;
}`

const softmaxClassifier = `// 3 sınıflı bir sınıflandırıcının çıkış katmanı
Matrix logits = hidden * outputWeights;   // ham skorlar
Matrix probs  = logits.softmax();          // olasılıklar

probs.download();
probs.print();   // her satır toplamı 1.0`

const trainingLoop = `// Ağırlıkları kaydet / yükle ile basit kalıcılık
Matrix W = Matrix::randn(256, 256);

// ... eğitim adımları ...

W.save("layer1.bin");                 // diske yaz
Matrix restored = Matrix::load("layer1.bin");  // geri yükle`

const linalgDemo = `// Doğrusal cebir: determinant ve ters matris
Matrix a{{4.0f, 7.0f},
         {2.0f, 6.0f}};

float det = a.determinant();   // 10.0
Matrix inv = a.inverse();      // tersini al

// Doğrulama: a * inv birim matrise yakın olmalı
Matrix check = a * inv;
check.download();
check.print();`

const examples = [
  {
    id: 'dense',
    title: 'Tam Bağlı Katman (Dense Layer)',
    desc: 'Matris çarpımı, bias yayınlama ve ReLU aktivasyonunu tek zincirde birleştiren temel bir sinir ağı katmanı.',
    code: linearLayer,
    tags: ['matmul', 'broadcast_add', 'relu'],
  },
  {
    id: 'softmax',
    title: 'Softmax Sınıflandırıcı',
    desc: 'Ham skorları (logits) sayısal kararlı bir olasılık dağılımına dönüştüren çıkış katmanı.',
    code: softmaxClassifier,
    tags: ['softmax'],
  },
  {
    id: 'persist',
    title: 'Ağırlık Kaydetme & Yükleme',
    desc: 'Eğitilmiş bir matrisi ikili biçimde diske yazma ve daha sonra geri yükleme.',
    code: trainingLoop,
    tags: ['save', 'load', 'randn'],
  },
  {
    id: 'linalg',
    title: 'Determinant & Ters Matris',
    desc: 'İleri doğrusal cebir operasyonlarıyla determinant hesaplama ve ters matris doğrulaması.',
    code: linalgDemo,
    tags: ['determinant', 'inverse'],
  },
]

export default function OrneklerPage() {
  return (
    <article>
      <PageHeader
        eyebrow="Kaynaklar"
        title="Örnekler"
        description="Gerçek dünya senaryolarında MatrixFlash-Pro'nun nasıl kullanıldığını gösteren, kopyalanmaya hazır kod parçaları."
      />

      <Callout type="tip" title="Çalıştırma">
        Bu örnekleri <InlineCode>examples/</InlineCode> klasörüne ekleyip{' '}
        <InlineCode>cmake --build build --config Release</InlineCode> ile
        derledikten sonra doğrudan çalıştırabilirsiniz.
      </Callout>

      <div className="mt-4 space-y-12">
        {examples.map((ex) => (
          <DocSection key={ex.id} id={ex.id} title={ex.title}>
            <p>{ex.desc}</p>
            <div className="mb-4 flex flex-wrap gap-2">
              {ex.tags.map((t) => (
                <span
                  key={t}
                  className="rounded-md border border-border bg-secondary/50 px-2 py-1 font-mono text-xs text-primary"
                >
                  {t}
                </span>
              ))}
            </div>
            <CodeBlock code={ex.code} filename={`${ex.id}.cpp`} />
          </DocSection>
        ))}
      </div>

      <div className="mt-8 rounded-2xl border border-primary/20 bg-primary/[0.04] p-6 text-center">
        <p className="font-semibold text-foreground">
          Daha fazla örnek arıyor musun?
        </p>
        <p className="mt-1 text-sm text-muted-foreground">
          Deponun <InlineCode>examples/</InlineCode> ve{' '}
          <InlineCode>tests/</InlineCode> klasörlerinde kütüphanenin tüm
          yeteneklerini gösteren daha fazla kod bulabilirsin.
        </p>
        <a
          href="https://github.com/muhamedsahin/MatrixFlash_Pro"
          target="_blank"
          rel="noreferrer"
          className="mt-4 inline-flex items-center gap-2 rounded-lg bg-primary px-5 py-2.5 text-sm font-semibold text-primary-foreground transition-transform hover:scale-[1.02]"
        >
          GitHub deposunu aç
        </a>
      </div>
    </article>
  )
}
