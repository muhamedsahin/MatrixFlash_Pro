'use client'

import Link from 'next/link'
import { ArrowRight, BookOpen, Terminal } from 'lucide-react'
import { GithubIcon } from '@/components/github-icon'
import { MatrixRain } from '@/components/matrix-rain'
import { Reveal } from '@/components/reveal'
import { site } from '@/lib/site'
import { useLanguage } from '@/lib/language-context'

export function Cta() {
  const { lang, t } = useLanguage()

  return (
    <section className="relative py-24 border-t border-border/80">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <Reveal>
          <div className="relative overflow-hidden rounded-3xl border border-primary/30 bg-gradient-to-b from-card/80 via-card/40 to-card/80 px-6 py-16 text-center shadow-2xl backdrop-blur-2xl sm:px-16">
            <div className="absolute inset-0 -z-0 opacity-20" aria-hidden>
              <MatrixRain className="h-full w-full" opacity={0.3} />
            </div>
            <div
              className="pointer-events-none absolute left-1/2 top-1/2 -z-0 h-96 w-96 -translate-x-1/2 -translate-y-1/2 rounded-full bg-primary/20 blur-[120px]"
              aria-hidden
            />
            <div className="relative">
              <h2 className="mx-auto max-w-2xl text-balance text-3xl font-extrabold tracking-tight sm:text-4xl lg:text-5xl">
                {t(
                  'Matris İşlemlerini GPU Hızına Taşımaya Hazır mısın?',
                  'Ready to Supercharge Your Matrix Workloads on GPU?',
                )}
              </h2>
              <p className="mx-auto mt-4 max-w-xl text-pretty text-base text-muted-foreground sm:text-lg">
                {t(
                  'C++17 ve CUDA 12.3+ ile sıfırdan yazılmış, cuBLAS ve Autograd destekli modern matris motoru.',
                  'Built from scratch in modern C++17 and CUDA 12.3+, powered by cuBLAS and an Autograd tape engine.',
                )}
              </p>
              <div className="mt-8 flex flex-wrap items-center justify-center gap-3.5">
                <Link
                  href="/docs"
                  className="glow-primary inline-flex items-center gap-2 rounded-xl bg-primary px-6 py-3.5 text-sm font-bold text-primary-foreground transition-all duration-200 hover:scale-[1.03]"
                >
                  <Terminal className="size-4" />
                  {t('Hemen Başlayın', 'Get Started')}
                  <ArrowRight className="size-4" />
                </Link>
                <Link
                  href="/docs/matris-nedir"
                  className="inline-flex items-center gap-2 rounded-xl border border-primary/30 bg-primary/10 px-6 py-3.5 text-sm font-semibold text-primary transition-all duration-200 hover:bg-primary/20"
                >
                  <BookOpen className="size-4" />
                  {t('Matris Derslerini İncele', 'Explore Matrix Course')}
                </Link>
                <a
                  href={site.github}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-2 rounded-xl border border-border/80 bg-secondary/50 px-6 py-3.5 text-sm font-semibold text-foreground transition-all duration-200 hover:bg-secondary"
                >
                  <GithubIcon className="size-4 text-primary" />
                  GitHub Repository
                </a>
              </div>
            </div>
          </div>
        </Reveal>
      </div>
    </section>
  )
}
