'use client'

import {
  Gauge,
  GitBranch,
  Layers,
  ShieldCheck,
  Zap,
  Activity,
  Hash,
  Eye,
  Cpu,
} from 'lucide-react'
import { Reveal } from '@/components/reveal'
import { useLanguage } from '@/lib/language-context'

export function Features() {
  const { lang, t } = useLanguage()

  const features = [
    {
      icon: Gauge,
      title: {
        tr: 'Shape-aware GEMM · cuBLAS-sınıfı',
        en: 'Shape-aware GEMM · cuBLAS-class',
      },
      body: {
        tr: 'Mikro / GEMV / algo-cache’li cublasLt / TENSOR_OP. 1024²’de cuBLAS’ın %99’u; 2048²’de geçer. `multiply_into` tahsissiz hot path.',
        en: 'Micro / GEMV / algo-cached cublasLt / TENSOR_OP. 99% of cuBLAS at 1024²; ahead at 2048². `multiply_into` is the allocation-free hot path.',
      },
      tag: 'Compute',
    },
    {
      icon: GitBranch,
      title: {
        tr: 'Autograd Ters-Mod Motoru',
        en: 'Autograd Reverse-Mode Tape Engine',
      },
      body: {
        tr: 'PyTorch benzeri hesaplama bandı (tape). Variable ve VarTensor sınıfları ile ileri ve geri yayılım doğrudan GPU belleğinde çalışır.',
        en: 'PyTorch-inspired computational graph tape. Variable and VarTensor objects run forward and backward passes directly in VRAM.',
      },
      tag: 'Deep Learning',
    },
    {
      icon: Cpu,
      title: {
        tr: 'cuSOLVER İleri Düzey Cebir',
        en: 'cuSOLVER Advanced Linear Algebra',
      },
      body: {
        tr: 'SVD (Tekil Değer Ayrışımı), QR, Cholesky, simetrik özdeğer/özvektör ve Moore-Penrose sözde ters (pinv) GPU ile hızlandırılır.',
        en: 'SVD, QR, Cholesky factorization, symmetric eigenvalues/eigenvectors, and pseudo-inverse (pinv) accelerated on CUDA.',
      },
      tag: 'Linalg',
    },
    {
      icon: Activity,
      title: {
        tr: 'Fused GEMM + Bias + Act',
        en: 'Fused GEMM + Bias + Act',
      },
      body: {
        tr: '`gemm_bias_relu` / `gemm_bias_gelu` tek cublasLt epilogue. Ayrı 3 kernel zincirine göre 1–6×. Ayrıca `fused_bias_gelu` elementwise zincirleri.',
        en: '`gemm_bias_relu` / `gemm_bias_gelu` as one cublasLt epilogue. 1–6× over a 3-kernel chain. Plus elementwise chains like `fused_bias_gelu`.',
      },
      tag: 'Optimization',
    },
    {
      icon: Hash,
      title: {
        tr: 'Seyrek Matrisler (Sparse CSR & SpMV)',
        en: 'Sparse CSR Matrices & SpMV',
      },
      body: {
        tr: 'Graf modelleri ve embedding tabloları için sıkıştırılmış satır (CSR) desteği. Seyrek x yoğun matris çarpımı ve seyreklik analizi.',
        en: 'Compressed Sparse Row format for graph models and embedding tables, supporting spmv and sparse-dense matrix multiplication.',
      },
      tag: 'Memory',
    },
    {
      icon: Eye,
      title: {
        tr: 'Sıfır Kopyalama Görünümleri (View)',
        en: 'Zero-Copy Strided MatrixView',
      },
      body: {
        tr: 'Adım (stride) takasıyla anında transpoz, alt matris dilimleme (slice) ve yeniden boyutlandırma — sıfır bayt bellek kopyalaması.',
        en: 'Instant transpose, sub-matrix slicing, and reshaping via stride manipulation without copying a single byte of VRAM.',
      },
      tag: 'Zero-Copy',
    },
    {
      icon: Zap,
      title: {
        tr: 'Stream Havuzu & Asenkron İndirgeme',
        en: 'Stream Pool & Async Reductions',
      },
      body: {
        tr: 'kStreamPoolSize = 4 ile bağımsız GPU işlerini eş zamanlı yürütme; argmax_async ile CPU iş parçacığını bloklamayan asenkron sonuçlar.',
        en: 'Concurrent multi-stream execution via pool_stream(), plus non-blocking argmax_async posting to host-pinned memory.',
      },
      tag: 'Streams',
    },
    {
      icon: ShieldCheck,
      title: {
        tr: 'Fail-Fast Stale-Mirror Güvenliği',
        en: 'Fail-Fast Stale-Mirror Safety',
      },
      body: {
        tr: 'GPU üzerinde değişen veriye download() yapılmadan erişilirse anında istisna fırlatır; sessiz veri bozulmalarını tamamen engeller.',
        en: 'Throws immediately if host accessors (at, data) are invoked on modified device buffers before download(), preventing silent bugs.',
      },
      tag: 'Safety',
    },
  ]

  return (
    <section className="relative py-24 border-t border-border/80">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <Reveal className="max-w-3xl">
          <div className="inline-flex items-center gap-2 font-mono text-xs uppercase tracking-[0.2em] text-primary">
            <Layers className="size-3.5" />
            <span>{t('NEDEN MATRIXFLASH-PRO?', 'WHY MATRIXFLASH-PRO?')}</span>
          </div>
          <h2 className="mt-4 text-balance text-3xl font-extrabold tracking-tight sm:text-4xl lg:text-5xl">
            {t(
              'Endüstri Standardında GPU Mimarisi, Sade C++17 Arayüzü',
              'Industry-Grade GPU Architecture, Elegant C++17 API',
            )}
          </h2>
          <p className="mt-4 text-pretty text-base text-muted-foreground sm:text-lg">
            {t(
              'cuBLAS, cuSOLVER, derin öğrenme operasyonları ve donanım dostu optimizasyonların modern bir hibrit birleşimi.',
              'A modern hybrid fusion of cuBLAS, cuSOLVER, deep learning primitives, and hardware-friendly CUDA optimizations.',
            )}
          </p>
        </Reveal>

        {/* Bento Grid */}
        <div className="mt-14 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {features.map((f, i) => (
            <Reveal
              key={i}
              delay={i * 40}
              className="group relative flex flex-col justify-between overflow-hidden rounded-2xl border border-border/80 bg-gradient-to-b from-card/90 to-card/50 p-6 transition-all duration-300 hover:border-primary/50 hover:shadow-[0_0_25px_rgba(110,231,183,0.12)] hover:-translate-y-0.5"
            >
              {/* Background radial glow */}
              <div className="pointer-events-none absolute -right-12 -top-12 size-36 rounded-full bg-primary/5 blur-2xl transition-all duration-300 group-hover:bg-primary/15" />

              <div>
                <div className="flex items-center justify-between">
                  <div className="flex size-11 items-center justify-center rounded-xl bg-primary/10 text-primary border border-primary/25 transition-colors group-hover:bg-primary/20">
                    <f.icon className="size-5" />
                  </div>
                  <span className="rounded-full border border-border/70 bg-secondary/50 px-2.5 py-0.5 font-mono text-[10px] text-muted-foreground">
                    {f.tag}
                  </span>
                </div>

                <h3 className="mt-5 font-mono text-base font-bold text-foreground group-hover:text-primary transition-colors">
                  {f.title[lang]}
                </h3>
                <p className="mt-2.5 text-xs leading-relaxed text-muted-foreground">
                  {f.body[lang]}
                </p>
              </div>

              <div className="mt-6 pt-3 border-t border-border/50 flex items-center justify-between text-[11px] font-mono text-primary/80">
                <span>CUDA Native</span>
                <span className="opacity-0 transition-opacity duration-200 group-hover:opacity-100">→</span>
              </div>
            </Reveal>
          ))}
        </div>
      </div>
    </section>
  )
}
