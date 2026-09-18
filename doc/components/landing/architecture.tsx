'use client'

import { FileCode2, FolderTree, CheckCircle2 } from 'lucide-react'
import { Reveal } from '@/components/reveal'
import { useLanguage } from '@/lib/language-context'

export function Architecture() {
  const { lang, t } = useLanguage()

  const tree = [
    {
      path: 'include/matrix_pro/core/',
      desc: { tr: 'Matrix, Tensor, MemoryMode, DType, MatrixProError', en: 'Matrix, Tensor, MemoryMode, DType, MatrixProError' },
      depth: 1,
    },
    {
      path: 'include/matrix_pro/ops/',
      desc: { tr: 'cuBLAS GEMM, 12+ indirgeme, yayınlama, cuSOLVER', en: 'cuBLAS GEMM, 12+ reductions, broadcasting, cuSOLVER' },
      depth: 1,
    },
    {
      path: 'include/matrix_pro/nn/',
      desc: { tr: 'Aktivasyonlar, NCHW Conv2D, Fused & In-place zincirleri', en: 'Activations, NCHW Conv2D, Fused & In-place chains' },
      depth: 1,
    },
    {
      path: 'include/matrix_pro/autograd/',
      desc: { tr: 'Variable & VarTensor ters-mod otomatik türev motoru', en: 'Variable & VarTensor reverse-mode autograd tape engine' },
      depth: 1,
    },
    {
      path: 'include/matrix_pro/sparse/',
      desc: { tr: 'Seyrek CSR matrisler, SpMV & seyrek çarpım', en: 'Sparse CSR matrices, SpMV & sparse-dense matmul' },
      depth: 1,
    },
    {
      path: 'include/matrix_pro/indexing/',
      desc: { tr: 'Gather, scatter-add ve embedding tabloları', en: 'Gather, scatter-add and embedding lookup tables' },
      depth: 1,
    },
    {
      path: 'include/matrix_pro/view/',
      desc: { tr: 'Sıfır kopyalama adımlı MatrixView (transpose, slice)', en: 'Zero-copy strided MatrixView (transpose, slice)' },
      depth: 1,
    },
    {
      path: 'include/matrix_pro/rng/',
      desc: { tr: 'Sayaç tabanlı deterministik cihaz RNG (MurmurHash3)', en: 'Counter-based deterministic device RNG (MurmurHash3)' },
      depth: 1,
    },
    {
      path: 'include/matrix_pro/streams/',
      desc: { tr: 'Stream havuzu (kStreamPoolSize=4) & asenkron indirgeme', en: 'Stream pool (kStreamPoolSize=4) & async reductions' },
      depth: 1,
    },
    {
      path: 'src/nn/conv.cu',
      desc: { tr: 'NCHW 2D konvolüsyon ve pooling CUDA kernel’ları', en: 'NCHW 2D convolution and pooling CUDA kernels' },
      depth: 0,
    },
    {
      path: 'src/operations_sparse.cu',
      desc: { tr: 'cuSPARSE ve GPU CSR birleşik operasyonları', en: 'cuSPARSE and GPU CSR coalesced operations' },
      depth: 0,
    },
  ]

  return (
    <section className="relative border-y border-border/80 bg-card/20 py-24">
      <div className="mx-auto grid max-w-7xl gap-14 px-4 sm:px-6 lg:grid-cols-2 lg:items-center lg:px-8">
        <Reveal>
          <div className="inline-flex items-center gap-2 font-mono text-xs uppercase tracking-[0.2em] text-primary">
            <FolderTree className="size-3.5" />
            <span>{t('TEMİZ MİMARİ', 'CLEAN ARCHITECTURE')}</span>
          </div>
          <h2 className="mt-4 text-balance text-3xl font-extrabold tracking-tight sm:text-4xl">
            {t(
              'Tek Sorumluluk İlkesiyle Ayrıştırılmış Modüler C++17',
              'Single-Responsibility Modular C++17 Architecture',
            )}
          </h2>
          <p className="mt-4 text-pretty leading-relaxed text-muted-foreground">
            {t(
              'MatrixFlash-Pro, devasa monolitik başlık dosyaları yerine her alt sistemin kendi header ve CUDA çeviri birimine sahip olduğu modüler bir yapıda tasarlanmıştır.',
              'MatrixFlash-Pro eschews monolithic files in favor of granular headers and dedicated CUDA translation units for each subsystem.',
            )}
          </p>
          <p className="mt-3 text-pretty leading-relaxed text-muted-foreground">
            {t(
              'Kullanıcılar tek bir umbrella başlık ("matrix_pro/matrix_pro.hpp") ile tüm kütüphaneyi içerebilir veya derleme süresini düşürmek için yalnızca ihtiyaç duydukları alt modülü dahil edebilir.',
              'Users can include the umbrella header ("matrix_pro/matrix_pro.hpp") for convenience or include granular headers directly to minimize compile times.',
            )}
          </p>

          <div className="mt-8 flex flex-wrap items-center gap-4 text-xs font-mono text-muted-foreground">
            <span className="flex items-center gap-1.5 text-foreground">
              <CheckCircle2 className="size-4 text-primary" />
              10+ Bağımsız Modül
            </span>
            <span className="flex items-center gap-1.5 text-foreground">
              <CheckCircle2 className="size-4 text-primary" />
              cuBLAS / cuSOLVER / cuSPARSE
            </span>
            <span className="flex items-center gap-1.5 text-foreground">
              <CheckCircle2 className="size-4 text-primary" />
              Header-Only Ops + CUDA .cu
            </span>
          </div>
        </Reveal>

        <Reveal delay={100}>
          <div className="overflow-hidden rounded-2xl border border-border/80 bg-[oklch(0.12_0.008_160)] shadow-2xl">
            <div className="flex items-center justify-between border-b border-border/70 bg-card/60 px-4 py-3 font-mono text-xs text-muted-foreground">
              <div className="flex items-center gap-2">
                <FolderTree className="size-3.5 text-primary" />
                <span className="text-foreground font-semibold">MatrixFlash_Pro/</span>
              </div>
              <span className="text-[10px] text-primary">C++17 / CUDA 12.3+</span>
            </div>

            <ul className="divide-y divide-border/40 font-mono text-xs">
              {tree.map((node, i) => (
                <li
                  key={i}
                  className="flex items-center gap-3 px-4 py-2.5 transition-colors hover:bg-white/[0.02]"
                  style={{ paddingLeft: `${16 + node.depth * 14}px` }}
                >
                  <FileCode2 className="size-3.5 shrink-0 text-primary/80" />
                  <code className="text-foreground font-medium">{node.path}</code>
                  <span className="ml-auto hidden truncate text-[11px] text-muted-foreground sm:block max-w-[240px]">
                    {node.desc[lang]}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        </Reveal>
      </div>
    </section>
  )
}
