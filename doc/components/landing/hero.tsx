'use client'

import Link from 'next/link'
import { ArrowRight, BookOpen, Cpu, Sparkles, Layers, Terminal } from 'lucide-react'
import { GithubIcon } from '@/components/github-icon'
import { MatrixRain } from '@/components/matrix-rain'
import { Hero3D } from '@/components/landing/hero-3d'
import { site } from '@/lib/site'
import { useLanguage } from '@/lib/hooks/use-language'

export function Hero() {
  const { t } = useLanguage()

  return (
    <section className="relative overflow-hidden pt-28 pb-10 sm:pt-36">
      {/* Background: animated matrix rain + grid + radial glow */}
      <div className="absolute inset-0 grid-bg-animated radial-fade" aria-hidden />
      <div className="absolute inset-0 -z-10 opacity-25" aria-hidden>
        <MatrixRain className="h-full w-full" opacity={0.16} />
      </div>
      <div
        className="pointer-events-none absolute left-1/2 top-0 -z-10 h-[600px] w-[1000px] -translate-x-1/2 rounded-full bg-primary/10 blur-[140px]"
        aria-hidden
      />

      <div className="relative mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="grid items-center gap-10 lg:grid-cols-12">
          {/* Left column: headline */}
          <div className="animate-fade-up lg:col-span-6">
            <div className="inline-flex items-center gap-2 rounded-full border border-primary/40 bg-primary/10 px-3.5 py-1.5 font-mono text-xs font-semibold text-primary shadow-[0_0_20px_rgba(110,231,183,0.2)]">
              <Sparkles className="size-3.5" />
              <span>v{site.version} · TF32 · Autograd · cuSOLVER</span>
            </div>

            <h1 className="fluid-display mt-6 text-balance">
              {t('GPU Hızlandırmalı', 'GPU-Accelerated')}{' '}
              <span className="bg-gradient-to-r from-primary via-emerald-300 to-accent bg-clip-text text-transparent animate-gradient-x">
                Matrix
              </span>{' '}
              {t('Kütüphanesi — C++17 ile', 'Library — in Modern C++17')}
            </h1>

            <p className="mt-5 max-w-xl text-pretty text-lg leading-relaxed text-muted-foreground">
              {t(
                'cuBLAS GEMM, TF32 Tensor Core, autograd tape motoru, cuSOLVER doğrusal cebir, seyrek CSR matrisleri ve asenkron stream havuzu — hepsi tek akıcı C++ arayüzünde.',
                'cuBLAS GEMM, TF32 Tensor Cores, an autograd tape engine, cuSOLVER linalg, sparse CSR matrices and an async stream pool — all behind one fluent C++ API.',
              )}
            </p>

            <div className="mt-8 flex flex-wrap items-center gap-4">
              <Link
                href="/docs"
                className="group inline-flex items-center gap-2 rounded-xl bg-primary px-6 py-3 text-sm font-semibold text-primary-foreground shadow-[0_0_30px_rgba(74,222,128,0.35)] transition-all hover:scale-[1.03] hover:shadow-[0_0_45px_rgba(74,222,128,0.5)]"
              >
                <Terminal className="size-4" />
                {t('Kuruluma Başla', 'Start Installation')}
                <ArrowRight className="size-4 transition-transform group-hover:translate-x-1" />
              </Link>
              <Link
                href="/docs/matris-nedir"
                className="inline-flex items-center gap-2 rounded-xl border border-border bg-card/60 px-6 py-3 text-sm font-semibold text-foreground transition-colors hover:bg-card hover:glow-primary"
              >
                <BookOpen className="size-4 text-primary" />
                {t('Matris Dersleri', 'Matrix Course')}
              </Link>
              <a
                href={site.github}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-2 px-2 py-3 font-mono text-xs text-muted-foreground transition-colors hover:text-primary"
              >
                <GithubIcon className="size-4" />
                GitHub v{site.version}
              </a>
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
