'use client'

import React, { useState } from 'react'
import {
  Boxes,
  Cpu,
  Zap,
  Activity,
  GitBranch,
  Eye,
  Hash,
  Layers,
  ArrowRight,
  Code2,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { useLanguage } from '@/lib/language-context'
import { CodeBlock } from '@/components/code-block'

interface ModuleInfo {
  id: string
  name: string
  header: string
  icon: React.ComponentType<{ className?: string }>
  badge: string
  title: { tr: string; en: string }
  desc: { tr: string; en: string }
  features: { tr: string[]; en: string[] }
  code: string
}

const MODULES: ModuleInfo[] = [
  {
    id: 'core',
    name: 'Core Module',
    header: 'matrix_pro/core/matrix.hpp',
    icon: Boxes,
    badge: 'C++17 / CUDA',
    title: {
      tr: 'Çekirdek Bellek & Tensör Mimarisi',
      en: 'Core Memory & Tensor Architecture',
    },
    desc: {
      tr: 'Host-Device ikili tamponu, MemoryMode::device_only ile sıfır gereksiz PCIe trafiği ve fail-fast stale-mirror güvenliği.',
      en: 'Dual host-device mirror, MemoryMode::device_only eliminating unwanted PCIe traffic, and fail-fast stale mirror safety.',
    },
    features: {
      tr: [
        'Matrix(M, N, MemoryMode::device_only) desteği',
        'Fail-fast: download() yapılmadan erişilirse istisna fırlatır',
        'Tensor rank-N veri yapısı (CNN & batched ops)',
        'Unified MatrixProError istisna hiyerarşisi',
      ],
      en: [
        'Matrix(M, N, MemoryMode::device_only) allocation',
        'Fail-fast stale-mirror contract (throws on invalid host access)',
        'Rank-N Tensor container for CNN and batched ops',
        'Unified MatrixProError exception hierarchy',
      ],
    },
    code: `// Device-resident matris: host kopyası oluşturulmaz
Matrix w(1024, 1024, MemoryMode::device_only);
w.fill(1.0f);

// GPU üzerinde işlem yürütülür
Matrix out = w * w;

// Host tarafında okumak için download() zorunludur (fail-fast)
out.download();
std::cout << "out(0,0): " << out.at(0, 0) << std::endl;`,
  },
  {
    id: 'ops',
    name: 'cuBLAS & Ops',
    header: 'matrix_pro/ops/operations.hpp',
    icon: Zap,
    badge: 'cuBLAS + TF32',
    title: {
      tr: 'cuBLAS GEMM & 12+ GPU İndirgemesi',
      en: 'cuBLAS GEMM & 12+ GPU Reductions',
    },
    desc: {
      tr: 'NVIDIA cuBLAS ve TF32 Tensor Core destekli matris çarpımı, eksen ve küresel istatistiksel indirgemeler.',
      en: 'NVIDIA cuBLAS accelerated GEMM with TF32 Tensor Cores, plus full global and axis reductions.',
    },
    features: {
      tr: [
        'cuBLAS SGEMM ile donanım sınırlarında TFLOPS',
        'sum, mean, min, max, argmin, argmax, variance, stddev',
        'l1_norm, l2_norm, frobenius_norm, determinant, trace',
        'Satır ve sütun bazlı yayınlama (broadcasting)',
      ],
      en: [
        'cuBLAS SGEMM reaching theoretical hardware TFLOPS',
        'sum, mean, min, max, argmin, argmax, variance, stddev',
        'l1_norm, l2_norm, frobenius_norm, determinant, trace',
        'Row and column vector broadcasting (add_row_vector, broadcast_add)',
      ],
    },
    code: `Matrix a = Matrix::randn(512, 512);
Matrix b = Matrix::randn(512, 512);

// cuBLAS GEMM hızlandırması
Matrix c = a * b;

// GPU üzerinde tek geçişte indirgemeler
float s = c.sum();
float mu = c.mean();
std::size_t best = c.argmax();`,
  },
  {
    id: 'autograd',
    name: 'Autograd Tape',
    header: 'matrix_pro/autograd/autograd.hpp',
    icon: GitBranch,
    badge: 'Automatic Diff',
    title: {
      tr: 'Ters-Mod Otomatik Türev Motoru',
      en: 'Reverse-Mode Automatic Differentiation',
    },
    desc: {
      tr: 'PyTorch benzeri hesaplama grafiği (computational graph tape). Geri yayılım (backprop) doğrudan GPU üzerinde çalışır.',
      en: 'PyTorch-style computational graph tape. Reverse-mode backward passes run directly on the GPU.',
    },
    features: {
      tr: [
        'Variable ve VarTensor sınıfları (requires_grad)',
        'backward() çağrısı ile zincir kuralı (Chain Rule) işletimi',
        'Custom unary ve custom loss fonksiyonu tanımlama imkanı',
        'Yayınlama ve birleştirme (concat) geri yayılım desteği',
      ],
      en: [
        'Variable and VarTensor abstractions (requires_grad)',
        'Automatic Chain Rule backpropagation via backward()',
        'Support for user-defined custom unary and custom loss ops',
        'Broadcasting and concatenation backward gradient dispatch',
      ],
    },
    code: `Variable x(Matrix::randn(32, 64), /*requires_grad=*/true);
Variable w(Matrix::randn(64, 10), /*requires_grad=*/true);
Variable b(Matrix::zeros(1, 10),  /*requires_grad=*/true);

// İleri yayılım (Forward pass)
Variable y = x.matmul(w).broadcast_add(b).relu();
Variable loss = mse_loss(y, target);

// Geri yayılım (Backward pass)
loss.backward();

// Gradyanlar GPU üzerinde hazır!
const Matrix& grad_w = w.grad();`,
  },
  {
    id: 'fused',
    name: 'Fused & In-place',
    header: 'matrix_pro/nn/fused.hpp',
    icon: Activity,
    badge: 'Memory Bandwidth',
    title: {
      tr: 'Birleştirilmiş (Fused) & Yerinde (In-Place) Kernel’lar',
      en: 'Fused Chains & In-Place CUDA Kernels',
    },
    desc: {
      tr: 'Tek bir GPU global bellek gidiş-dönüşü ile birden fazla matematiksel işlemi birleştiren yüksek verimli çekirdekler.',
      en: 'High-throughput kernels collapsing multiple operations into a single global memory round-trip.',
    },
    features: {
      tr: [
        'relu_, sigmoid_, gelu_, add_ in-place varyantları',
        'fused_sigmoid_mul: sigmoid(x) * y (LSTM / Gated gate)',
        'fused_bias_gelu: Transformer MLP katmanı için ideal',
        'Ara bellek tahsislerini sıfıra indirir',
      ],
      en: [
        'In-place ops: relu_, sigmoid_, gelu_, add_, broadcast_add_',
        'fused_sigmoid_mul: sigmoid(x) * y (Gated activation)',
        'fused_bias_gelu: Optimized for Transformer MLP blocks',
        'Eliminates intermediate buffer allocations completely',
      ],
    },
    code: `Matrix x = Matrix::randn(256, 1024);
Matrix bias = Matrix::zeros(256, 1024);

// 2 işlem yerine tek GPU kernel'ı çalışır!
Matrix activated = fused_bias_gelu(x, bias);

// Yerinde (in-place) bellek tasarrufu:
add_(activated, 0.01f);
relu_(activated);`,
  },
  {
    id: 'cusolver',
    name: 'cuSOLVER Linalg',
    header: 'matrix_pro/ops/linalg.hpp',
    icon: Cpu,
    badge: 'NVIDIA cuSOLVER',
    title: {
      tr: 'İleri Düzey Doğrusal Cebir (cuSOLVER)',
      en: 'Advanced Linear Algebra with cuSOLVER',
    },
    desc: {
      tr: 'QR ayrışımı, Tekil Değer Ayrışımı (SVD), Cholesky, Özdeğer/Özvektör ve sözde ters (pinv) hesaplamaları.',
      en: 'QR decomposition, SVD, Cholesky factorization, Eigenvalues/Eigenvectors, and Moore-Penrose pseudo-inverse.',
    },
    features: {
      tr: [
        'SVD: Tekil değerler ve U, V matrisleri',
        'Cholesky: Pozitif tanımlı kovaryans matrisi çarpanlarına ayırma',
        'Eigen: Simetrik matrisler için özvektör ve özdeğer analizi',
        'solve_least_squares: Aşırı belirlenmiş denklem çözümü',
      ],
      en: [
        'SVD: Singular values with U and Vt orthogonal matrices',
        'Cholesky: LL^T factorization for positive definite matrices',
        'Eigen: Symmetric eigenvalue and eigenvector decomposition',
        'solve_least_squares: High-precision pseudo-inverse fitting',
      ],
    },
    code: `Matrix a{{4.0f, 1.0f}, {1.0f, 3.0f}};

// cuSOLVER Tekil Değer Ayrışımı
SVDResult svd = a.svd();
// svd.u, svd.s (tekil değerler), svd.vt

// Cholesky ve Özdeğer
Matrix l = a.cholesky();
EigenResult eig = a.eigen();`,
  },
  {
    id: 'conv',
    name: '2D Conv & Pooling',
    header: 'matrix_pro/nn/conv.hpp',
    icon: Layers,
    badge: 'NCHW Tensor',
    title: {
      tr: '2D Konvolüsyon ve Havuzlama (CNN)',
      en: '2D Convolution & Pooling (CNN)',
    },
    desc: {
      tr: 'NCHW formatında ileri ve geri yayılım destekli 2B konvolüsyon, MaxPool ve AvgPool çekirdekleri.',
      en: 'Forward and backward differentiable 2D convolution, max pooling, and average pooling in NCHW format.',
    },
    features: {
      tr: [
        'conv2d(input, weights, bias, stride, padding)',
        'conv2d_input_backward, conv2d_weight_backward gradyanları',
        'max_pool2d ve avg_pool2d katmanları',
        'cuDNN uyumlu hızlı yol',
      ],
      en: [
        'conv2d(input, weights, bias, stride, padding)',
        'conv2d_input_backward, conv2d_weight_backward gradients',
        'max_pool2d and avg_pool2d pooling layers',
        'cuDNN backend fallback integration',
      ],
    },
    code: `// [Batch=16, Kanallar=3, Yükseklik=32, Genişlik=32]
Tensor x({16, 3, 32, 32});
Tensor w({64, 3, 3, 3}); // 64 filtre
Tensor b({64});

// Konvolüsyon + MaxPool
Tensor feat = conv2d(x, w, b, /*stride=*/1, /*padding=*/1);
Tensor pooled = max_pool2d(feat, /*kernel_size=*/2, /*stride=*/2);`,
  },
  {
    id: 'sparse',
    name: 'Sparse CSR',
    header: 'matrix_pro/sparse/sparse.hpp',
    icon: Hash,
    badge: 'cuSPARSE',
    title: {
      tr: 'Seyrek Matrisler (Sparse CSR & SpMV)',
      en: 'Compressed Sparse Row (CSR & SpMV)',
    },
    desc: {
      tr: 'Graf sinir ağları ve büyük dil modellerinde embedding tabloları için optimize edilmiş CSR bellek formatı.',
      en: 'CSR format optimized for graph neural networks, large embedding tables, and sparse-dense matrix operations.',
    },
    features: {
      tr: [
        'from_dense() veya COO koordinat listesinden oluşturma',
        'spmv: Seyrek matris-vektör çarpımı',
        'sparse_matmul: Seyrek x Yoğun matris çarpımı',
        'Bellek kullanımını seyreklik oranına göre katbekat düşürür',
      ],
      en: [
        'Construct from dense matrix or explicit COO triplets',
        'spmv: Sparse matrix-vector multiply (y = A * x)',
        'sparse_matmul: Sparse-dense GEMM',
        'Drastically reduces VRAM footprint based on sparsity',
      ],
    },
    code: `Matrix dense = Matrix::zeros(10000, 10000);
// Birkaç elemanı doldur...
SparseCSR sparse = SparseCSR::from_dense(dense, /*threshold=*/1e-5f);

Matrix x = Matrix::randn(10000, 1);
// Seyrek matris-vektör çarpımı
Matrix y = spmv(sparse, x);`,
  },
  {
    id: 'view',
    name: 'Zero-Copy Views',
    header: 'matrix_pro/view/view.hpp',
    icon: Eye,
    badge: 'Zero-Copy',
    title: {
      tr: 'Sıfır Kopyalama Görünümleri (MatrixView)',
      en: 'Zero-Copy Strided MatrixView',
    },
    desc: {
      tr: 'VRAM üzerinde tek bir bayt dahi kopyalamadan adım (stride) manipülasyonu ile transpoz, dilimleme ve yeniden boyutlandırma.',
      en: 'Manipulate strides without copying a single byte in VRAM for instant transpose, slicing, and reshaping.',
    },
    features: {
      tr: [
        'transpose_view: Adımları takas ederek anında devrik alma',
        'slice_view: Alt matris üzerinde sıfır bellek maliyetli işlem',
        'reshape_view: Bitişik matrisler için anında boyut değişimi',
        'materialize: Gerektiğinde tek kopyayla Matrix nesnesine dönüştürme',
      ],
      en: [
        'transpose_view: Instant transposition by swapping strides',
        'slice_view: Sub-matrix slicing with zero memory allocation',
        'reshape_view: Non-copy reshaping for contiguous buffers',
        'materialize: Materialize strided views back to dense Matrix',
      ],
    },
    code: `Matrix m = Matrix::randn(1024, 2048);

// 0 kopyalama, 0 VRAM tahsisi!
MatrixView v = transpose_view(m); // 2048 x 1024
MatrixView patch = slice_view(m, 0, 128, 0, 128);

// İhtiyaç duyulduğunda yoğun matrise dönüştür:
Matrix dense_patch = materialize(patch);`,
  },
]

export function ModuleExplorer() {
  const [selectedId, setSelectedId] = useState<string>('core')
  const { lang, t } = useLanguage()

  const currentMod = MODULES.find((m) => m.id === selectedId) || MODULES[0]

  return (
    <div className="relative rounded-2xl border border-border/80 bg-gradient-to-b from-card/80 to-card/40 p-6 shadow-2xl backdrop-blur-xl">
      {/* Header */}
      <div className="flex flex-col gap-2 border-b border-border/60 pb-5 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <div className="inline-flex items-center gap-2 font-mono text-xs text-primary">
            <Code2 className="size-3.5" />
            <span>{t('MODÜLER C++17 MİMARİSİ', 'MODULAR C++17 ARCHITECTURE')}</span>
          </div>
          <h3 className="mt-1 text-xl font-bold tracking-tight text-foreground sm:text-2xl">
            {t('MatrixFlash-Pro Modül Gezgini', 'MatrixFlash-Pro Module Explorer')}
          </h3>
        </div>
        <p className="max-w-md text-xs text-muted-foreground sm:text-right">
          {t(
            'İlgilendiğiniz modülü seçerek başlık dosyasını, yeteneklerini ve C++ kod örneğini canlı inceleyin.',
            'Select any module to inspect its header file, key features, and live C++ code snippet.',
          )}
        </p>
      </div>

      {/* Module Selector Pill Tabs */}
      <div className="mt-6 flex flex-wrap gap-2">
        {MODULES.map((mod) => {
          const isSelected = mod.id === selectedId
          const Icon = mod.icon
          return (
            <button
              key={mod.id}
              type="button"
              onClick={() => setSelectedId(mod.id)}
              className={cn(
                'group flex items-center gap-2 rounded-xl border px-3.5 py-2 font-mono text-xs font-semibold transition-all duration-200',
                isSelected
                  ? 'border-primary/60 bg-primary/15 text-primary shadow-[0_0_15px_rgba(110,231,183,0.25)]'
                  : 'border-border/60 bg-secondary/30 text-muted-foreground hover:border-border hover:bg-secondary/60 hover:text-foreground',
              )}
            >
              <Icon className={cn('size-3.5', isSelected ? 'text-primary' : 'text-muted-foreground')} />
              <span>{mod.name}</span>
            </button>
          )
        })}
      </div>

      {/* Module Detail Card */}
      <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-12 lg:items-stretch">
        {/* Features Column */}
        <div className="flex flex-col justify-between rounded-xl border border-border/80 bg-[oklch(0.12_0.008_160)] p-5 lg:col-span-5">
          <div>
            <div className="flex items-center justify-between gap-2">
              <span className="rounded-md border border-primary/30 bg-primary/10 px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider text-primary">
                {currentMod.badge}
              </span>
              <span className="font-mono text-[11px] text-muted-foreground/80">
                #{currentMod.id}
              </span>
            </div>

            <h4 className="mt-3 font-mono text-base font-bold text-foreground">
              {currentMod.title[lang]}
            </h4>
            <p className="mt-2 text-xs leading-relaxed text-muted-foreground">
              {currentMod.desc[lang]}
            </p>

            <div className="mt-5 space-y-2 border-t border-border/50 pt-4">
              <span className="font-mono text-[10px] uppercase tracking-wider text-primary/80 font-bold">
                {t('Öne Çıkan Yetenekler', 'Key Capabilities')}
              </span>
              <ul className="space-y-1.5">
                {currentMod.features[lang].map((f, i) => (
                  <li key={i} className="flex items-start gap-2 text-xs text-muted-foreground">
                    <span className="mt-1 size-1 rounded-full bg-primary shrink-0" />
                    <span>{f}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          <div className="mt-5 rounded-lg border border-border/50 bg-black/30 p-2.5 font-mono text-[11px] text-accent flex items-center justify-between">
            <span className="truncate">#include &quot;{currentMod.header}&quot;</span>
            <span className="shrink-0 text-[10px] text-muted-foreground">CUDA include</span>
          </div>
        </div>

        {/* Code Column */}
        <div className="lg:col-span-7 flex flex-col">
          <CodeBlock code={currentMod.code} filename={currentMod.header} />
        </div>
      </div>
    </div>
  )
}

