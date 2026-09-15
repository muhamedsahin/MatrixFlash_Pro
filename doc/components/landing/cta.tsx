import Link from 'next/link'
import { ArrowRight } from 'lucide-react'
import { GithubIcon } from '@/components/github-icon'
import { MatrixRain } from '@/components/matrix-rain'
import { Reveal } from '@/components/reveal'
import { site } from '@/lib/site'

export function Cta() {
  return (
    <section className="relative py-24">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <Reveal>
          <div className="relative overflow-hidden rounded-3xl border border-primary/20 bg-card/40 px-6 py-16 text-center sm:px-16">
            <div className="absolute inset-0 -z-0 opacity-20" aria-hidden>
              <MatrixRain className="h-full w-full" opacity={0.5} />
            </div>
            <div
              className="pointer-events-none absolute left-1/2 top-1/2 -z-0 h-80 w-80 -translate-x-1/2 -translate-y-1/2 rounded-full bg-primary/15 blur-[100px]"
              aria-hidden
            />
            <div className="relative">
              <h2 className="mx-auto max-w-2xl text-balance text-3xl font-bold tracking-tight sm:text-4xl">
                Matris işlemlerini GPU&apos;ya taşımaya hazır mısın?
              </h2>
              <p className="mx-auto mt-4 max-w-xl text-pretty text-muted-foreground">
                Kurulumdan API referansına, matris matematiğinden gerçek
                örneklere kadar her şey dokümantasyonda.
              </p>
              <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
                <Link
                  href="/docs"
                  className="glow-primary inline-flex items-center gap-2 rounded-lg bg-primary px-6 py-3 text-sm font-semibold text-primary-foreground transition-transform hover:scale-[1.02]"
                >
                  Dokümantasyona Git
                  <ArrowRight className="size-4" />
                </Link>
                <a
                  href={site.github}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-2 rounded-lg border border-border bg-secondary/50 px-6 py-3 text-sm font-semibold transition-colors hover:bg-secondary"
                >
                  <GithubIcon className="size-4" />
                  Kaynak Kod
                </a>
              </div>
            </div>
          </div>
        </Reveal>
      </div>
    </section>
  )
}
