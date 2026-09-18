'use client'

import React, { useState } from 'react'
import Link from 'next/link'
import { ArrowRight, Cpu, Zap, Terminal, Sparkles, BookOpen, Layers } from 'lucide-react'
import { GithubIcon } from '@/components/github-icon'
import { MatrixRain } from '@/components/matrix-rain'
import { CodeBlock } from '@/components/code-block'
import { site } from '@/lib/site'
import { useLanguage } from '@/lib/language-context'
import { cn } from '@/lib/utils'

const QUICKSTART_SNIPPETS = {
  basic: {
    label: 'Quickstart GEMM',
    filename: 'quickstart.cpp',
    code: `#include "matrix_pro/matrix_pro.hpp"
#include <iostream>

int main() {
    using matrix_pro::Matrix;
    using matrix_pro::MemoryMode;

    // Device-resident matris (PCIe yükü yok)
    Matrix a{{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix b = Matrix::identity(2);

    // cuBLAS GEMM + ReLU aktivasyonu
    Matrix c = (a * b).relu();

    // Fail-fast kuralı: download() çağrılmadan host okuması yapılamaz
    c.download();
    std::cout << "c(0,0): " << c.at(0, 0) << "\\n";
    return 0;
}`,
  },
  autograd: {
    label: 'Autograd Tape',
    filename: 'autograd_demo.cpp',
    code: `#include "matrix_pro/matrix_pro.hpp"

using matrix_pro::Variable;
using matrix_pro::Matrix;

int main() {
    // Otomatik türev değişkenleri
    Variable x(Matrix::randn(64, 128), /*requires_grad=*/true);
    Variable w(Matrix::randn(128, 10), /*requires_grad=*/true);

    // İleri yayılım: Y = x * W
    Variable y = x.matmul(w).relu();

    // Geri yayılım (Backpropagation)
    y.backward();

    // x ve w için gradyanlar GPU üzerinde hazır!
    const Matrix& grad_w = w.grad();
    return 0;
}`,
  },
  fused: {
    label: 'Fused & In-Place',
    filename: 'fused_perf.cpp',
    code: `#include "matrix_pro/matrix_pro.hpp"

int main() {
    using namespace matrix_pro;

    Matrix x = Matrix::uniform(512, 512, -1.0f, 1.0f);
    Matrix bias = Matrix::zeros(512, 512);

    // Tek GPU kernel'ında bias + GeLU birleşimi
    Matrix activated = fused_bias_gelu(x, bias);

    // Bellek tasarruflu yerinde (in-place) işlem
    relu_(activated);
    add_(activated, 0.5f);
    return 0;
}`,
  },
}

export function Hero() {
  const { lang, t } = useLanguage()
  const [activeTab, setActiveTab] = useState<keyof typeof QUICKSTART_SNIPPETS>('basic')

  return (
    <section className="relative overflow-hidden pt-28 pb-20 sm:pt-36">
      {/* Background Matrix Rain + Radial Blur */}
      <div className="absolute inset-0 grid-bg radial-fade" aria-hidden />
      <div className="absolute inset-0 -z-10 opacity-30" aria-hidden>
        <MatrixRain className="h-full w-full" opacity={0.18} />
      </div>
      <div
        className="pointer-events-none absolute left-1/2 top-0 -z-10 h-[600px] w-[1000px] -translate-x-1/2 rounded-full bg-primary/10 blur-[140px]"
        aria-hidden
      />

      <div className="relative mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="grid items-center gap-12 lg:grid-cols-12">
          {/* Left Column: Headline & Value Proposition */}
          <div className="animate-fade-up lg:col-span-7">
            {/* Holographic Badge */}
            <div className="inline-flex items-center gap-2 rounded-full border border-primary/40 bg-primary/10 px-3.5 py-1.5 font-mono text-xs font-semibold text-primary shadow-[0_0_20px_rgba(110,231,183,0.2)]">
              <Zap className="size-3.5 text-primary animate-pulse" />
              <span>{site.cuda} · cuBLAS + TF32 · Autograd Tape</span>
            </div>

            {/* Main Headline */}
            <h1 className="mt-6 text-balance text-4xl font-extrabold leading-[1.08] tracking-tight sm:text-5xl lg:text-6xl">
              {lang === 'tr' ? (
                <>
                  GPU Hızında{' '}
                  <span className="bg-gradient-to-r from-emerald-400 via-primary to-cyan-400 bg-clip-text text-transparent text-glow">
                    Modern C++17
                  </span>{' '}
                  Matris & Tensör Motoru
                </>
              ) : (
                <>
                  GPU-Accelerated{' '}
                  <span className="bg-gradient-to-r from-emerald-400 via-primary to-cyan-400 bg-clip-text text-transparent text-glow">
                    Modern C++17
                  </span>{' '}
                  Matrix & Tensor Engine
                </>
              )}
            </h1>

            {/* Subheading */}
            <p className="mt-6 max-w-2xl text-pretty text-base leading-relaxed text-muted-foreground sm:text-lg">
              {lang === 'tr' ? (
                <>
                  <strong className="font-mono text-foreground">MatrixFlash-Pro</strong>,
                  düşük seviyeli CUDA kernel karmaşıklığını sıfırlayan;{' '}
                  <span className="text-primary font-semibold">cuBLAS GEMM</span>,{' '}
                  <span className="text-accent font-semibold">Autograd hesaplama grafiği</span>,{' '}
                  <span className="text-foreground font-semibold">cuSOLVER doğrusal cebir</span>,{' '}
                  seyrek (sparse) matrisler ve fail-fast bellek modeli sunan ultra performanslı bir kütüphanedir.
                </>
              ) : (
                <>
                  <strong className="font-mono text-foreground">MatrixFlash-Pro</strong> eliminates
                  low-level CUDA kernel boilerplate; delivering{' '}
                  <span className="text-primary font-semibold">cuBLAS GEMM</span>, an{' '}
                  <span className="text-accent font-semibold">Autograd reverse-mode tape</span>,{' '}
                  <span className="text-foreground font-semibold">cuSOLVER linalg</span>,{' '}
                  sparse CSR matrices, and fail-fast stale-mirror safety.
                </>
              )}
            </p>

            {/* Action Buttons */}
            <div className="mt-8 flex flex-wrap items-center gap-3.5">
              <Link
                href="/docs"
                className="glow-primary inline-flex items-center gap-2 rounded-xl bg-primary px-6 py-3.5 text-sm font-bold text-primary-foreground transition-all duration-200 hover:scale-[1.03] hover:shadow-[0_0_25px_rgba(110,231,183,0.5)]"
              >
                <Terminal className="size-4" />
                {t('Kütüphaneyi Keşfet', 'Explore Documentation')}
                <ArrowRight className="size-4" />
              </Link>

              <Link
                href="/docs/matris-nedir"
                className="inline-flex items-center gap-2 rounded-xl border border-primary/30 bg-primary/10 px-5 py-3.5 text-sm font-semibold text-primary transition-all duration-200 hover:bg-primary/20 hover:border-primary/60"
              >
                <BookOpen className="size-4" />
                {t('Matris Dersleri (Sıfırdan)', 'Matrix Math Course')}
              </Link>

              <a
                href={site.github}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-2 rounded-xl border border-border/80 bg-secondary/50 px-5 py-3.5 text-sm font-semibold text-foreground transition-all duration-200 hover:bg-secondary hover:border-border"
              >
                <GithubIcon className="size-4 text-primary" />
                <span className="font-mono text-xs">GitHub v{site.version}</span>
              </a>
            </div>

            {/* Hardware badges ticker */}
            <div className="mt-10 flex flex-wrap items-center gap-5 border-t border-border/60 pt-6 font-mono text-xs text-muted-foreground">
              <div className="flex items-center gap-2">
                <Cpu className="size-4 text-primary" />
                <span>NVIDIA CUDA 12.3+</span>
              </div>
              <span className="size-1 rounded-full bg-border" />
              <div className="flex items-center gap-2">
                <Layers className="size-4 text-accent" />
                <span>cuBLAS & cuSOLVER</span>
              </div>
              <span className="size-1 rounded-full bg-border" />
              <div className="flex items-center gap-2">
                <Sparkles className="size-4 text-emerald-400" />
                <span>{site.cpp} Standard</span>
              </div>
            </div>
          </div>

          {/* Right Column: Tabbed Code Showcase */}
          <div className="lg:col-span-5 animate-fade-up [animation-delay:150ms]">
            <div className="relative rounded-2xl border border-border/80 bg-gradient-to-b from-card to-card/60 p-2 shadow-2xl backdrop-blur-xl">
              {/* Tab Selector */}
              <div className="flex items-center justify-between border-b border-border/60 px-3 py-2">
                <div className="flex items-center gap-1.5">
                  <span className="size-3 rounded-full bg-destructive/60" />
                  <span className="size-3 rounded-full bg-amber-500/60" />
                  <span className="size-3 rounded-full bg-emerald-500/60" />
                </div>
                <div className="flex items-center gap-1">
                  {(Object.keys(QUICKSTART_SNIPPETS) as Array<keyof typeof QUICKSTART_SNIPPETS>).map(
                    (key) => (
                      <button
                        key={key}
                        type="button"
                        onClick={() => setActiveTab(key)}
                        className={cn(
                          'rounded-md px-2.5 py-1 font-mono text-[11px] font-semibold transition-all',
                          activeTab === key
                            ? 'bg-primary/20 text-primary border border-primary/40'
                            : 'text-muted-foreground hover:bg-white/[0.04] hover:text-foreground',
                        )}
                      >
                        {QUICKSTART_SNIPPETS[key].label}
                      </button>
                    ),
                  )}
                </div>
              </div>

              {/* Active Code Block */}
              <div className="p-2">
                <CodeBlock
                  code={QUICKSTART_SNIPPETS[activeTab].code}
                  filename={QUICKSTART_SNIPPETS[activeTab].filename}
                />
              </div>

              {/* Status footer bar */}
              <div className="flex items-center justify-between px-3 py-2 font-mono text-[10px] text-muted-foreground border-t border-border/50">
                <span className="flex items-center gap-1.5 text-primary">
                  <span className="size-1.5 rounded-full bg-primary animate-ping" />
                  GPU Memory: Resident
                </span>
                <span>nvcc -O3 -std=c++17</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
