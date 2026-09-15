import Link from 'next/link'
import { ArrowRight, Cpu, Zap } from 'lucide-react'
import { GithubIcon } from '@/components/github-icon'
import { MatrixRain } from '@/components/matrix-rain'
import { MatrixViz } from '@/components/matrix-viz'
import { CodeBlock } from '@/components/code-block'
import { site } from '@/lib/site'

const heroCode = `#include "matrix_pro/matrix.hpp"
using matrix_pro::Matrix;

Matrix input{{1.0f, -2.0f}, {3.0f, 4.0f}};
Matrix weights = Matrix::identity(2);

// GPU üzerinde zincirleme işlem
Matrix output = (input * weights).relu().softmax();
output.download();`

export function Hero() {
  return (
    <section className="relative overflow-hidden pt-32 pb-20 sm:pt-40">
      {/* backgrounds */}
      <div className="absolute inset-0 grid-bg radial-fade" aria-hidden />
      <div className="absolute inset-0 -z-0" aria-hidden>
        <MatrixRain className="h-full w-full" opacity={0.14} />
      </div>
      <div
        className="pointer-events-none absolute left-1/2 top-0 -z-0 h-[500px] w-[900px] -translate-x-1/2 rounded-full bg-primary/10 blur-[120px]"
        aria-hidden
      />

      <div className="relative mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="grid items-center gap-12 lg:grid-cols-2">
          <div className="animate-fade-up">
            <div className="inline-flex items-center gap-2 rounded-full border border-primary/30 bg-primary/10 px-3 py-1 font-mono text-xs text-primary">
              <Zap className="size-3.5" />
              CUDA kernels · CUDA {site.cuda}
            </div>

            <h1 className="mt-6 text-balance text-5xl font-bold leading-[1.05] tracking-tight sm:text-6xl lg:text-7xl">
              GPU hızında
              <br />
              <span className="text-primary text-glow">matris</span> işlemleri
            </h1>

            <p className="mt-6 max-w-xl text-pretty text-lg leading-relaxed text-muted-foreground">
              <span className="font-mono text-foreground">MatrixFlash-Pro</span>,
              CUDA kernel karmaşıklığını tamamen soyutlayarak sade ve akıcı bir{' '}
              <span className="font-mono text-primary">Matrix</span> arayüzü
              sunan, modern {site.cpp} ile yazılmış yüksek performanslı matris
              kütüphanesidir.
            </p>

            <div className="mt-8 flex flex-wrap items-center gap-3">
              <Link
                href="/docs"
                className="glow-primary inline-flex items-center gap-2 rounded-lg bg-primary px-5 py-3 text-sm font-semibold text-primary-foreground transition-transform hover:scale-[1.02]"
              >
                Başlayın
                <ArrowRight className="size-4" />
              </Link>
              <a
                href={site.github}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-2 rounded-lg border border-border bg-secondary/50 px-5 py-3 text-sm font-semibold transition-colors hover:bg-secondary"
              >
                <GithubIcon className="size-4" />
                GitHub&apos;da İncele
              </a>
            </div>

            <div className="mt-10 flex items-center gap-6 font-mono text-xs text-muted-foreground">
              <span className="flex items-center gap-2">
                <Cpu className="size-4 text-primary" /> NVIDIA GPU
              </span>
              <span className="h-4 w-px bg-border" />
              <span>{site.cpp}</span>
              <span className="h-4 w-px bg-border" />
              <span>tiled kernels · CMake</span>
            </div>
          </div>

          <div className="relative animate-fade-up [animation-delay:150ms]">
            <div className="relative">
              <CodeBlock code={heroCode} filename="quickstart.cpp" />
              <div className="absolute -right-4 -top-6 hidden animate-float lg:block">
                <MatrixViz rows={3} cols={3} label="device memory" />
              </div>
              <div className="absolute -bottom-8 -left-6 hidden animate-float [animation-delay:1.5s] lg:block">
                <MatrixViz rows={2} cols={4} label="softmax()" />
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
