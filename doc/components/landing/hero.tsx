'use client'

import Link from 'next/link'
import { ArrowRight, BookOpen, Cpu, Sparkles, Layers, Terminal } from 'lucide-react'
import { MatrixRain } from '@/components/matrix-rain'
import { Hero3D } from '@/components/landing/hero-3d'
import { site } from '@/lib/site'
import { useLanguage } from '@/lib/hooks/use-language'

export function Hero() {
  const { t } = useLanguage()

  return (
    <section className="relative overflow-hidden pt-28 pb-14 sm:pt-36">
      <div className="absolute inset-0 mesh-aurora" aria-hidden />
      <div className="absolute inset-0 grid-bg-animated radial-fade opacity-70" aria-hidden />
      <div className="absolute inset-0 -z-10 opacity-20" aria-hidden>
        <MatrixRain className="h-full w-full" opacity={0.14} />
      </div>
      <div
        className="pointer-events-none absolute left-1/2 top-[-10%] -z-10 h-[640px] w-[1100px] -translate-x-1/2 rounded-full bg-primary/12 blur-[150px]"
        aria-hidden
      />

      <div className="relative mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="grid items-center gap-12 lg:grid-cols-12">
          <div className="animate-fade-up lg:col-span-6">
            <div className="inline-flex items-center gap-2 rounded-full border border-primary/35 bg-primary/10 px-3.5 py-1.5 font-mono text-xs font-semibold text-primary shadow-[0_0_28px_oklch(0.84_0.2_145_/_0.22)]">
              <Sparkles className="size-3.5" />
              <span>v{site.version} · cuBLAS-class · Fused GEMM</span>
            </div>

            <h1 className="fluid-display mt-7 text-balance font-display">
              <span className="mb-3 block font-mono text-[0.42em] uppercase tracking-[0.08em] text-primary text-glow">
                MatrixFlash-Pro
              </span>
              {t('GPU’da dünya sınıfı', 'World-class on the GPU')}{' '}
              <span className="bg-gradient-to-r from-primary via-emerald-200 to-accent bg-clip-text text-transparent animate-gradient-x">
                {t('matris motoru', 'matrix engine')}
              </span>
            </h1>

            <p className="mt-5 max-w-xl text-pretty text-lg leading-relaxed text-muted-foreground">
              {t(
                'Shape-aware GEMM, TF32, fused bias+act, Autograd ve cuSOLVER — doğrulanmış: 1024², 2048² ve 4096²’de cuBLAS’ı geçiyor (16,895 GFLOPS tepe verim).',
                'Shape-aware GEMM, TF32, fused bias+act, Autograd and cuSOLVER — verified: beats cuBLAS at 1024², 2048², and 4096² (16,895 GFLOPS peak).',
              )}
            </p>

            <div className="mt-8 flex flex-wrap items-center gap-3">
              <Link
                href="/docs"
                className="group inline-flex items-center gap-2 rounded-2xl bg-primary px-6 py-3.5 text-sm font-semibold text-primary-foreground shadow-[0_0_36px_oklch(0.84_0.2_145_/_0.35)] transition-all hover:scale-[1.03] hover:shadow-[0_0_52px_oklch(0.84_0.2_145_/_0.5)]"
              >
                <Terminal className="size-4" />
                {t('Kuruluma Başla', 'Start Installation')}
                <ArrowRight className="size-4 transition-transform group-hover:translate-x-1" />
              </Link>
              <a
                href="/MatrixFlash_Pro_Technical_Manual.pdf"
                download="MatrixFlash_Pro_Technical_Manual.pdf"
                className="inline-flex items-center gap-2 rounded-2xl border border-emerald-500/40 bg-emerald-500/10 px-5 py-3.5 text-sm font-semibold text-emerald-400 hover:bg-emerald-500/20 hover:border-emerald-500/60 transition-all shadow-[0_0_20px_rgba(16,185,129,0.15)]"
              >
                <BookOpen className="size-4 text-emerald-400" />
                {t('Teknik Manuel (57s PDF)', 'Technical Manual (57p PDF)')}
              </a>
              <Link
                href="/docs/performans"
                className="inline-flex items-center gap-2 rounded-2xl border border-border/80 glass-panel px-5 py-3.5 text-sm font-semibold text-foreground transition-colors hover:border-primary/40"
              >
                <Cpu className="size-4 text-primary" />
                {t('Performans', 'Benchmarks')}
              </Link>
            </div>

            {/* Hardware ticker */}
            <div className="mt-10 flex flex-wrap items-center gap-5 border-t border-border/60 pt-6 font-mono text-xs text-muted-foreground">
              <div className="flex items-center gap-2">
                <Cpu className="size-4 text-primary" />
                <span>NVIDIA CUDA 12.3+</span>
              </div>
              <span className="size-1 rounded-full bg-border" />
              <div className="flex items-center gap-2">
                <Layers className="size-4 text-accent" />
                <span>cuBLAS · cuSOLVER · cuSPARSE</span>
              </div>
              <span className="size-1 rounded-full bg-border" />
              <div className="flex items-center gap-2">
                <Sparkles className="size-4 text-emerald-400" />
                <span>{site.cpp} Standard</span>
              </div>
            </div>
          </div>

          {/* Right column: interactive 3D matrix lattice */}
          <div className="animate-fade-up [animation-delay:150ms] lg:col-span-6">
            <Hero3D />
          </div>
        </div>
      </div>
    </section>
  )
}
